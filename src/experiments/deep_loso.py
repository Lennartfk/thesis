from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm
import torch

from src.data.eeg_dataset import ChannelStandardizer, infer_epoch_shape, load_epochs_from_index
from src.experiments.evaluate import binary_metrics, confusion_counts, safe_roc_auc
from src.experiments.loso import iter_loso_splits
from src.models.adaptation.adabn import adapt_batch_norm
from src.models.deep.eegnet import build_eegnet
from src.training.torch_trainer import predict_torch_model, train_torch_model
from src.tracking.mlflow_utils import log_torch_model


def choose_validation_subject(test_subject_id, subject_order, strategy="cyclic_next_subject"):
    if len(subject_order) < 2:
        raise ValueError("Need at least two subjects to create a subject-level validation split.")
    if strategy == "cyclic_next_subject":
        try:
            current_index = subject_order.index(str(test_subject_id))
        except ValueError as exc:
            raise ValueError(f"Test subject {test_subject_id} not found in subject order.") from exc
        return subject_order[(current_index + 1) % len(subject_order)]
    raise ValueError(f"Unknown validation subject strategy: {strategy}")


def build_deep_model(config, n_channels, n_samples):
    if config.model_name != "eegnet":
        raise ValueError("The deep LOSO runner currently supports model_name='eegnet'.")

    return build_eegnet(
        seed=config.seed,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=2,
        dropout=config.eegnet_dropout,
        f1=config.eegnet_f1,
        depth_multiplier=config.eegnet_depth_multiplier,
    )


def normalize_fold(X_train, X_val, X_test, normalization):
    if normalization == "none":
        return X_train, X_val, X_test
    if normalization != "channel_standard":
        raise ValueError(f"Unknown normalization: {normalization}")

    normalizer = ChannelStandardizer().fit(X_train)
    return normalizer.transform(X_train), normalizer.transform(X_val), normalizer.transform(X_test)


def save_checkpoint(output_dir, test_subject_id, model):
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"eegnet_subject_{test_subject_id}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_deep_loso_experiment(epoch_index, config, output_dir):
    if config.adaptation_name not in {"none", "adabn"}:
        raise ValueError("Deep adaptation supports adaptation_name='none' or 'adabn' for EEGNet.")

    n_channels, n_samples, sfreq, channel_names = infer_epoch_shape(config.epoch_dir)
    fold_rows = []
    prediction_rows = []
    history_rows = []
    checkpoint_paths = []
    start = perf_counter()
    
    # Count total folds for progress bar
    loso_splits = list(iter_loso_splits(epoch_index))
    total_folds = len(loso_splits)

    # Build ordered list of subjects to allow cyclic validation subject selection
    subject_order = [str(s) for s, _, _ in loso_splits]

    for fold_index, (subject_id, train_idx, test_idx) in tqdm(list(enumerate(loso_splits, start=1)), total=total_folds, desc="LOSO Folds", unit="fold"):
        test_subject_id = str(subject_id)
        outer_train_df = epoch_index.iloc[train_idx].reset_index(drop=True)
        test_df = epoch_index.iloc[test_idx].reset_index(drop=True)
        val_subject_id = choose_validation_subject(
            test_subject_id,
            subject_order,
            strategy=config.validation_subject_strategy,
        )

        val_mask = outer_train_df["subject_id"].astype(str) == str(val_subject_id)
        train_df = outer_train_df.loc[~val_mask].reset_index(drop=True)
        val_df = outer_train_df.loc[val_mask].reset_index(drop=True)

        X_train, y_train, _ = load_epochs_from_index(train_df)
        X_val, y_val, _ = load_epochs_from_index(val_df)
        X_test, y_test, test_metadata = load_epochs_from_index(test_df)
        X_train, X_val, X_test = normalize_fold(X_train, X_val, X_test, config.normalization)

        model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
        fold_start = perf_counter()
        model, history, training_info = train_torch_model(model, X_train, y_train, X_val, y_val, config)
        adaptation_info = {
            "adabn_applied": False,
            "adabn_batch_norm_layers": 0,
            "adabn_target_samples": 0,
            "adabn_target_batches": 0,
        }
        if config.adaptation_name == "adabn":
            adaptation_info = adapt_batch_norm(model, X_test, config)
        test_metrics, _, y_pred, y_score = predict_torch_model(model, X_test, y_test, config)

        metrics = binary_metrics(y_test, y_pred)
        metrics["roc_auc"] = safe_roc_auc(y_test, y_score)
        metrics.update(confusion_counts(y_test, y_pred))
        train_subjects = sorted(outer_train_df["subject_id"].astype(str).unique())

        metrics.update(
            {
                "fold": fold_index,
            "test_subject_id": test_subject_id,
            "val_subject_id": val_subject_id,
            "train_subjects": ",".join(train_subjects),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_alert": int((y_train == 0).sum()),
                "n_train_drowsy": int((y_train == 1).sum()),
                "n_val_alert": int((y_val == 0).sum()),
                "n_val_drowsy": int((y_val == 1).sum()),
                "n_test_alert": int((y_test == 0).sum()),
                "n_test_drowsy": int((y_test == 1).sum()),
                "fold_seconds": perf_counter() - fold_start,
                "adaptation_uses_target_unlabeled": config.adaptation_name == "adabn",
                "best_epoch": training_info["best_epoch"],
                "best_val_metric_name": training_info["best_val_metric_name"],
                "best_val_metric": training_info["best_val_metric"],
                "trained_epochs": training_info["trained_epochs"],
                "training_seconds": training_info["training_seconds"],
                "device": training_info["device"],
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sfreq": sfreq,
            }
        )
        metrics.update(adaptation_info)
        fold_rows.append(metrics)

        history = history.copy()
        history["fold"] = fold_index
        history["test_subject_id"] = test_subject_id
        history["val_subject_id"] = val_subject_id
        history_rows.append(history)

        predictions = test_metadata[["subject_id", "recording", "epoch_index", "perclos", "target"]].copy()
        predictions["fold"] = fold_index
        predictions["y_pred"] = y_pred
        predictions["y_score"] = y_score
        prediction_rows.append(predictions)

        checkpoint_paths.append(save_checkpoint(output_dir, test_subject_id, model))
        log_torch_model(
            model,
            input_example=torch.zeros((1, n_channels, n_samples), dtype=torch.float32),
            artifact_path=f"models_fold_{fold_index:02d}",
        )

        print(
            f"[{fold_index:02d}] test_subject={test_subject_id} val={val_subject_id} "
            f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} best_epoch={training_info['best_epoch']}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    print(f"Finished deep LOSO in {perf_counter() - start:.1f}s")

    metadata = {
        "feature_count": int(n_channels * n_samples),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sfreq": sfreq,
        "channels": ",".join(channel_names),
        "checkpoint_paths": checkpoint_paths,
    }
    return fold_metrics, predictions, history, metadata
