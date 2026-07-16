from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm
import torch

from src.data.epochs import ChannelStandardizer, infer_epoch_shape, load_epochs_from_index
from src.experiments.evaluate import binary_metrics, confusion_counts, safe_roc_auc, select_binary_threshold
from src.experiments.loso import iter_within_subject_splits
from src.models.deep.eegnet import build_eegnet
from src.training.torch_trainer import predict_torch_model, train_torch_model
from src.tracking.mlflow_utils import log_torch_model


def build_deep_model(config, n_channels, n_samples):
    if config.model_name != "eegnet":
        raise ValueError("The deep within-subject runner currently supports model_name='eegnet'.")

    return build_eegnet(
        seed=config.seed,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=2,
        dropout=config.eegnet_dropout,
        f1=config.eegnet_f1,
        depth_multiplier=config.eegnet_depth_multiplier,
        temporal_kernel_length=config.eegnet_temporal_kernel_length,
        separable_kernel_length=config.eegnet_separable_kernel_length,
        pool1_kernel=config.eegnet_pool1_kernel,
        pool2_kernel=config.eegnet_pool2_kernel,
    )


def normalize_fold(X_train, X_val, X_test, normalization):
    if normalization == "none":
        return X_train, X_val, X_test
    if normalization != "channel_standard":
        raise ValueError(f"Unknown normalization: {normalization}")

    normalizer = ChannelStandardizer().fit(X_train)
    return normalizer.transform(X_train), normalizer.transform(X_val), normalizer.transform(X_test)


def save_checkpoint(output_dir, subject_id, fold_num, model):
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"eegnet_subject_{subject_id}_fold_{fold_num}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_deep_within_subject_experiment(epoch_index, config, output_dir):
    """
    Within-subject experiment: for each subject, split their epochs into train/test.
    This provides an oracle upper bound on performance (same subject, no domain shift).
    """
    if config.adaptation_name != "none":
        raise ValueError("Deep adaptation is not implemented yet; use adaptation_name='none'.")

    n_channels, n_samples, sfreq, channel_names = infer_epoch_shape(config.epoch_dir)
    fold_rows = []
    prediction_rows = []
    history_rows = []
    checkpoint_paths = []
    start = perf_counter()

    # Count total folds for progress bar
    within_subject_splits = list(
        iter_within_subject_splits(
            epoch_index,
            strategy=config.within_subject_split_strategy,
            seed=config.seed,
        )
    )
    total_folds = len(within_subject_splits)

    for fold_index, (subject_id, train_idx, val_idx, test_idx) in tqdm(
        list(enumerate(within_subject_splits, start=1)), total=total_folds, desc="Within-Subject Folds", unit="fold"
    ):
        test_subject_id = str(subject_id)
        train_df = epoch_index.iloc[train_idx].reset_index(drop=True)
        val_df = epoch_index.iloc[val_idx].reset_index(drop=True)
        test_df = epoch_index.iloc[test_idx].reset_index(drop=True)

        X_train, y_train, _ = load_epochs_from_index(train_df)
        X_val, y_val, _ = load_epochs_from_index(val_df)
        X_test, y_test, test_metadata = load_epochs_from_index(test_df)
        X_train, X_val, X_test = normalize_fold(X_train, X_val, X_test, config.normalization)

        model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
        fold_start = perf_counter()
        model, history, training_info = train_torch_model(model, X_train, y_train, X_val, y_val, config)
        _, _, _, y_val_score = predict_torch_model(
            model,
            X_val,
            y_val,
            config,
            decision_threshold=config.decision_threshold,
        )
        if config.tune_decision_threshold:
            decision_threshold, threshold_val_metric, threshold_val_metrics = select_binary_threshold(
                y_val,
                y_val_score,
                metric=config.threshold_metric,
                default_threshold=config.decision_threshold,
            )
        else:
            decision_threshold = config.decision_threshold
            threshold_val_metric = None
            threshold_val_metrics = {}

        test_metrics, _, y_pred, y_score = predict_torch_model(
            model,
            X_test,
            y_test,
            config,
            decision_threshold=decision_threshold,
        )

        metrics = binary_metrics(y_test, y_pred)
        metrics["roc_auc"] = safe_roc_auc(y_test, y_score)
        metrics.update(confusion_counts(y_test, y_pred))
        n_pred_alert = int((y_pred == 0).sum())
        n_pred_drowsy = int((y_pred == 1).sum())

        metrics.update(
            {
                "fold": fold_index,
                "subject_id": test_subject_id,
                "test_subject_id": test_subject_id,
                "val_subject_id": test_subject_id,
                "train_subjects": test_subject_id,
                "within_subject_split_strategy": config.within_subject_split_strategy,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_alert": int((y_train == 0).sum()),
                "n_train_drowsy": int((y_train == 1).sum()),
                "n_val_alert": int((y_val == 0).sum()),
                "n_val_drowsy": int((y_val == 1).sum()),
                "n_test_alert": int((y_test == 0).sum()),
                "n_test_drowsy": int((y_test == 1).sum()),
                "n_pred_alert": n_pred_alert,
                "n_pred_drowsy": n_pred_drowsy,
                "true_drowsy_rate": float((y_test == 1).mean()),
                "pred_drowsy_rate": float((y_pred == 1).mean()),
                "fold_seconds": perf_counter() - fold_start,
                "best_epoch": training_info["best_epoch"],
                "best_val_metric_name": training_info["best_val_metric_name"],
                "best_val_metric": training_info["best_val_metric"],
                "decision_threshold": decision_threshold,
                "threshold_tuned": config.tune_decision_threshold,
                "threshold_metric": config.threshold_metric,
                "threshold_val_metric": threshold_val_metric,
                "threshold_val_accuracy": threshold_val_metrics.get("accuracy"),
                "threshold_val_balanced_accuracy": threshold_val_metrics.get("balanced_accuracy"),
                "threshold_val_precision": threshold_val_metrics.get("precision"),
                "threshold_val_recall": threshold_val_metrics.get("recall"),
                "threshold_val_f1": threshold_val_metrics.get("f1"),
                "trained_epochs": training_info["trained_epochs"],
                "training_seconds": training_info["training_seconds"],
                "device": training_info["device"],
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sfreq": sfreq,
            }
        )
        fold_rows.append(metrics)

        history = history.copy()
        history["fold"] = fold_index
        history["subject_id"] = test_subject_id
        history["test_subject_id"] = test_subject_id
        history["val_subject_id"] = test_subject_id
        history_rows.append(history)

        predictions = test_metadata[["subject_id", "recording", "epoch_index", "perclos", "target"]].copy()
        predictions["fold"] = fold_index
        predictions["y_pred"] = y_pred
        predictions["y_score"] = y_score
        prediction_rows.append(predictions)

        if config.save_checkpoints:
            checkpoint_paths.append(save_checkpoint(output_dir, test_subject_id, fold_index, model))
        if config.log_models:
            log_torch_model(
                model,
                input_example=torch.zeros((1, n_channels, n_samples), dtype=torch.float32),
                artifact_path=f"models_fold_{fold_index:02d}",
            )

        print(
            f"[{fold_index:02d}] subject={test_subject_id} "
            f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} threshold={decision_threshold:.3f} "
            f"best_epoch={training_info['best_epoch']}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    print(f"Finished within-subject experiment in {perf_counter() - start:.1f}s")

    metadata = {
        "feature_count": int(n_channels * n_samples),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sfreq": sfreq,
        "channels": ",".join(channel_names),
        "checkpoint_paths": checkpoint_paths,
    }
    return fold_metrics, predictions, history, metadata
