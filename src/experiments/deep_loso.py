import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.data.epochs import ChannelStandardizer, infer_epoch_shape, load_epochs_from_index
from src.experiments.evaluate import binary_metrics, confusion_counts, safe_roc_auc, select_binary_threshold
from src.experiments.loso import iter_loso_splits
from src.models.adaptation.adabn import adapt_batch_norm
from src.models.adaptation.euclidean_alignment import align_subjects
from src.models.deep.eegnet import build_eegnet
from src.training.torch_trainer import predict_torch_model, train_torch_model
from src.tracking.mlflow_utils import log_torch_model


def choose_validation_subject(test_subject_id, subject_order, candidate_df=None, strategy="cyclic_next_subject", seed=None, min_samples_per_class=0):
    test_subject_id = str(test_subject_id)
    if len(subject_order) < 2:
        raise ValueError("Need at least two subjects to create a subject-level validation split.")
    try:
        current_index = subject_order.index(test_subject_id)
    except ValueError as exc:
        raise ValueError(f"Test subject {test_subject_id} not found in subject order.") from exc

    candidate_subjects = [s for s in subject_order if s != test_subject_id]
    
    valid_candidates = []
    if min_samples_per_class > 0 and candidate_df is not None:
        for s in candidate_subjects:
            s_mask = candidate_df["subject_id"].astype(str) == s
            s_targets = candidate_df.loc[s_mask, "target"]
            if (s_targets == 0).sum() >= min_samples_per_class and (s_targets == 1).sum() >= min_samples_per_class:
                valid_candidates.append(s)
        
        if not valid_candidates:
            print(f"Warning: No validation subject has >= {min_samples_per_class} samples for both classes. Falling back to all candidates.")
            valid_candidates = candidate_subjects
    else:
        valid_candidates = candidate_subjects

    if strategy == "cyclic_next_subject":
        for i in range(1, len(subject_order)):
            candidate = subject_order[(current_index + i) % len(subject_order)]
            if candidate in valid_candidates:
                return candidate
        return valid_candidates[0]
    if strategy == "previous_subject":
        for i in range(1, len(subject_order)):
            candidate = subject_order[(current_index - i) % len(subject_order)]
            if candidate in valid_candidates:
                return candidate
        return valid_candidates[0]
    if strategy == "random_subject":
        if seed is None:
            raise ValueError("Random validation subject strategy requires a seed.")
        rng = random.Random(f"{seed}:{test_subject_id}")
        return rng.choice(valid_candidates)
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


def apply_euclidean_alignment(X, metadata):
    subjects = metadata["subject_id"].astype(str).to_numpy()
    return align_subjects(X, subjects)


def split_target_adaptation_data(X_test, y_test, test_metadata, adaptation_fraction, seed, test_subject_id):
    adaptation_fraction = float(adaptation_fraction)
    if adaptation_fraction <= 0.0 or adaptation_fraction > 1.0:
        raise ValueError("adaptation_target_fraction must be in the interval (0, 1].")

    n_total = len(X_test)
    if n_total == 0:
        raise ValueError(f"Target fold for subject {test_subject_id} has no samples.")

    if np.isclose(adaptation_fraction, 1.0):
        return {
            "X_adapt": X_test,
            "y_adapt": y_test,
            "metadata_adapt": test_metadata,
            "X_eval": X_test,
            "y_eval": y_test,
            "metadata_eval": test_metadata,
            "mode": "full_target",
            "n_adapt": int(n_total),
            "n_eval": int(n_total),
            "n_total": int(n_total),
        }

    if n_total < 2:
        raise ValueError(
            f"Target fold for subject {test_subject_id} needs at least 2 samples for fraction-based adaptation."
        )

    n_adapt = int(round(n_total * adaptation_fraction))
    n_adapt = max(1, min(n_adapt, n_total - 1))

    rng = random.Random(f"{seed}:{test_subject_id}:{adaptation_fraction:.6f}")
    indices = list(range(n_total))
    rng.shuffle(indices)
    adapt_indices = indices[:n_adapt]
    eval_indices = indices[n_adapt:]

    return {
        "X_adapt": X_test[adapt_indices],
        "y_adapt": y_test[adapt_indices],
        "metadata_adapt": test_metadata.iloc[adapt_indices].reset_index(drop=True),
        "X_eval": X_test[eval_indices],
        "y_eval": y_test[eval_indices],
        "metadata_eval": test_metadata.iloc[eval_indices].reset_index(drop=True),
        "mode": "held_out_remaining",
        "n_adapt": int(len(adapt_indices)),
        "n_eval": int(len(eval_indices)),
        "n_total": int(n_total),
    }


def save_checkpoint(output_dir, test_subject_id, model):
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"eegnet_subject_{test_subject_id}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_deep_loso_experiment(epoch_index, config, output_dir):
    if config.adaptation_name not in {"none", "adabn", "ea", "ea_adabn"}:
        raise ValueError("Deep adaptation supports adaptation_name='none', 'adabn', 'ea', or 'ea_adabn' for EEGNet.")

    n_channels, n_samples, sfreq, channel_names = infer_epoch_shape(config.epoch_dir)
    fold_rows = []
    prediction_rows = []
    history_rows = []
    checkpoint_paths = []
    start = perf_counter()

    # Count total folds for progress bar
    loso_splits = list(iter_loso_splits(epoch_index))
    total_folds = len(loso_splits)

    # Build ordered list of subjects to allow subject-level validation selection
    subject_order = [str(s) for s, _, _ in loso_splits]
    target_fraction = float(getattr(config, "adaptation_target_fraction", 1.0))

    for fold_index, (subject_id, train_idx, test_idx) in tqdm(
        list(enumerate(loso_splits, start=1)),
        total=total_folds,
        desc="LOSO Folds",
        unit="fold",
    ):
        test_subject_id = str(subject_id)
        outer_train_df = epoch_index.iloc[train_idx].reset_index(drop=True)
        test_df = epoch_index.iloc[test_idx].reset_index(drop=True)
        val_subject_id = choose_validation_subject(
            test_subject_id,
            subject_order,
            outer_train_df,
            strategy=config.validation_subject_strategy,
            seed=config.seed,
            min_samples_per_class=config.val_min_class_samples,
        )

        val_mask = outer_train_df["subject_id"].astype(str) == str(val_subject_id)
        train_df = outer_train_df.loc[~val_mask].reset_index(drop=True)
        val_df = outer_train_df.loc[val_mask].reset_index(drop=True)

        X_train, y_train, train_metadata = load_epochs_from_index(train_df)
        X_val, y_val, val_metadata = load_epochs_from_index(val_df)
        X_test, y_test, test_metadata = load_epochs_from_index(test_df)
        target_split = split_target_adaptation_data(
            X_test,
            y_test,
            test_metadata,
            adaptation_fraction=target_fraction,
            seed=config.seed,
            test_subject_id=test_subject_id,
        )
        ea_info = {
            "ea_applied": False,
            "ea_train_subjects": 0,
            "ea_val_subjects": 0,
            "ea_target_subjects": 0,
            "ea_target_samples": 0,
        }
        if config.adaptation_name in {"ea", "ea_adabn"}:
            X_train, train_aligner = apply_euclidean_alignment(X_train, train_metadata)
            X_val, val_aligner = apply_euclidean_alignment(X_val, val_metadata)
            if target_split["mode"] == "full_target":
                X_test, test_aligner = apply_euclidean_alignment(target_split["X_eval"], target_split["metadata_eval"])
                target_split["X_adapt"] = X_test
            else:
                X_adapt_aligned, test_aligner = apply_euclidean_alignment(target_split["X_adapt"], target_split["metadata_adapt"])
                target_split["X_adapt"] = X_adapt_aligned
                X_test = test_aligner.transform(
                    target_split["X_eval"],
                    target_split["metadata_eval"]["subject_id"].astype(str).to_numpy(),
                )
            ea_info = {
                "ea_applied": True,
                "ea_train_subjects": len(train_aligner.transforms_),
                "ea_val_subjects": len(val_aligner.transforms_),
                "ea_target_subjects": len(test_aligner.transforms_),
                "ea_target_samples": int(target_split["n_adapt"]),
                "adaptation_target_fraction": target_fraction,
                "adaptation_target_mode": target_split["mode"],
                "adaptation_target_total_samples": target_split["n_total"],
                "adaptation_target_eval_samples": int(len(X_test)),
            }
            # For EA, X_test is already the aligned X_eval.
            # We must set target_split["X_eval"] to X_test so it gets normalized.
            target_split["X_eval"] = X_test

        if config.normalization != "none":
            normalizer = ChannelStandardizer().fit(X_train)
            X_train = normalizer.transform(X_train)
            X_val = normalizer.transform(X_val)
            X_test = normalizer.transform(X_test)
            target_split["X_adapt"] = normalizer.transform(target_split["X_adapt"])
            target_split["X_eval"] = normalizer.transform(target_split["X_eval"])

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

        adaptation_info = {
            "adabn_applied": False,
            "adabn_batch_norm_layers": 0,
            "adabn_target_samples": 0,
            "adabn_target_batches": 0,
        }
        if config.adaptation_name in {"adabn", "ea_adabn"}:
            adaptation_info = adapt_batch_norm(model, target_split["X_adapt"], config)
            adaptation_info.update(
                {
                    "adabn_target_samples": int(target_split["n_adapt"]),
                    "adaptation_target_fraction": target_fraction,
                    "adaptation_target_mode": target_split["mode"],
                    "adaptation_target_total_samples": target_split["n_total"],
                    "adaptation_target_eval_samples": int(target_split["n_eval"]),
                }
            )
        # Evaluate using target_split["X_eval"] which is correctly aligned and normalized.
        X_eval = target_split["X_eval"]
        y_eval = target_split["y_eval"]
        test_metrics, _, y_pred, y_score = predict_torch_model(
            model,
            X_eval,
            y_eval,
            config,
            decision_threshold=decision_threshold,
        )

        metrics = binary_metrics(y_eval, y_pred)
        metrics["roc_auc"] = safe_roc_auc(y_eval, y_score)
        metrics.update(confusion_counts(y_eval, y_pred))
        n_pred_alert = int((y_pred == 0).sum())
        n_pred_drowsy = int((y_pred == 1).sum())
        train_subjects = sorted(outer_train_df["subject_id"].astype(str).unique())

        metrics.update(
            {
                "fold": fold_index,
                "test_subject_id": test_subject_id,
                "val_subject_id": val_subject_id,
                "train_subjects": ",".join(train_subjects),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(X_eval),
                "n_target_total": target_split["n_total"],
                "n_target_adapt": target_split["n_adapt"],
                "n_target_eval": target_split["n_eval"],
                "n_train_alert": int((y_train == 0).sum()),
                "n_train_drowsy": int((y_train == 1).sum()),
                "n_val_alert": int((y_val == 0).sum()),
                "n_val_drowsy": int((y_val == 1).sum()),
                "n_test_alert": int((y_eval == 0).sum()),
                "n_test_drowsy": int((y_eval == 1).sum()),
                "n_pred_alert": n_pred_alert,
                "n_pred_drowsy": n_pred_drowsy,
                "true_drowsy_rate": float((y_eval == 1).mean()),
                "pred_drowsy_rate": float((y_pred == 1).mean()),
                "fold_seconds": perf_counter() - fold_start,
                "adaptation_uses_target_unlabeled": config.adaptation_name in {"adabn", "ea", "ea_adabn"},
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
        metrics.update(adaptation_info)
        metrics.update(ea_info)
        fold_rows.append(metrics)

        history = history.copy()
        history["fold"] = fold_index
        history["test_subject_id"] = test_subject_id
        history["val_subject_id"] = val_subject_id
        history_rows.append(history)

        predictions = target_split["metadata_eval"][["subject_id", "recording", "epoch_index", "perclos", "target"]].copy()
        predictions["fold"] = fold_index
        predictions["y_pred"] = y_pred
        predictions["y_score"] = y_score
        prediction_rows.append(predictions)

        if config.save_checkpoints:
            checkpoint_paths.append(save_checkpoint(output_dir, test_subject_id, model))
        if config.log_models:
            log_torch_model(
                model,
                input_example=torch.zeros((1, n_channels, n_samples), dtype=torch.float32),
                artifact_path=f"models_fold_{fold_index:02d}",
            )

        print(
            f"[{fold_index:02d}] test_subject={test_subject_id} val={val_subject_id} "
            f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} threshold={decision_threshold:.3f} "
            f"best_epoch={training_info['best_epoch']}"
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
