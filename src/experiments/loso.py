from time import perf_counter

import numpy as np
import pandas as pd

from src.experiments.evaluate import binary_metrics, confusion_counts, safe_roc_auc, try_predict_scores
from src.experiments.registry import get_adaptation, get_model
from src.tracking.mlflow_utils import log_sklearn_model


def subject_sort_key(subject_id):
    subject_text = str(subject_id)
    try:
        return (0, int(subject_text))
    except ValueError:
        return (1, subject_text)


def iter_loso_splits(df, subject_column="subject_id"):
    for subject_id in sorted(df[subject_column].astype(str).unique(), key=subject_sort_key):
        test_mask = df[subject_column].astype(str) == subject_id
        yield subject_id, np.flatnonzero(~test_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())


def iter_within_subject_splits(
    df,
    train_fraction=0.70,
    val_fraction=0.15,
    test_fraction=0.15,
    subject_column="subject_id",
    time_column="epoch_index",
    target_column="target",
    strategy="stratified",
    seed=42,
):
    """
    Within-subject split: for each subject, split their epochs into train/val/test.

    Args:
        df: DataFrame with subject_id column
        train_fraction: fraction of epochs used for training
        val_fraction: fraction of epochs used for validation
        test_fraction: fraction of epochs used for testing
        subject_column: name of subject ID column
        time_column: column used for chronological ordering when strategy="chronological"
        target_column: target label column used when strategy="stratified"
        strategy: "stratified" keeps every split class-balanced; "chronological" uses time order
        seed: random seed used by the stratified splitter
    """
    total_fraction = train_fraction + val_fraction + test_fraction
    if not np.isclose(total_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    if strategy not in {"stratified", "chronological"}:
        raise ValueError(f"Unknown within-subject split strategy: {strategy}")

    rng = np.random.default_rng(seed)

    def class_split_counts(n_samples):
        if n_samples < 3:
            return None

        counts = np.array([1, 1, 1], dtype=int)
        remaining = n_samples - int(counts.sum())
        fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
        raw_additions = remaining * fractions
        additions = np.floor(raw_additions).astype(int)
        counts += additions

        leftover = n_samples - int(counts.sum())
        if leftover > 0:
            order = np.argsort(-(raw_additions - additions))
            for split_index in order[:leftover]:
                counts[split_index] += 1

        return tuple(int(count) for count in counts)

    for subject_id in sorted(df[subject_column].astype(str).unique(), key=subject_sort_key):
        subject_mask = df[subject_column].astype(str) == subject_id
        subject_positions = np.flatnonzero(subject_mask.to_numpy())
        subject_df = df.iloc[subject_positions]
        if subject_df.empty:
            continue

        if strategy == "stratified":
            if subject_df[target_column].nunique() < 2:
                print(
                    f"[warn] skipping subject {subject_id}: within-subject binary evaluation "
                    "requires both alert and drowsy samples."
                )
                continue

            train_parts = []
            val_parts = []
            test_parts = []
            skip_subject = False

            for target in sorted(subject_df[target_column].unique()):
                class_positions = subject_positions[subject_df[target_column].to_numpy() == target]
                rng.shuffle(class_positions)

                split_counts = class_split_counts(len(class_positions))
                if split_counts is None:
                    print(
                        f"[warn] skipping subject {subject_id}: class {target} has only "
                        f"{len(class_positions)} retained samples; need at least 3."
                    )
                    skip_subject = True
                    break

                n_train_class, n_val_class, n_test_class = split_counts
                train_parts.append(class_positions[:n_train_class])
                val_parts.append(class_positions[n_train_class : n_train_class + n_val_class])
                test_parts.append(class_positions[n_train_class + n_val_class :])

            if skip_subject:
                continue

            train_idx = np.concatenate(train_parts)
            val_idx = np.concatenate(val_parts)
            test_idx = np.concatenate(test_parts)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)

            if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
                yield subject_id, train_idx, val_idx, test_idx
            continue

        subject_order = np.argsort(subject_df[time_column].to_numpy(), kind="stable")
        subject_indices = subject_positions[subject_order]

        if len(subject_indices) < 3:
            continue  # Skip subjects with too few epochs

        n_total = len(subject_indices)
        n_train = max(1, int(n_total * train_fraction))
        n_val = max(1, int(n_total * val_fraction))
        n_test = n_total - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1

        train_idx = subject_indices[:n_train]
        val_idx = subject_indices[n_train : n_train + n_val]
        test_idx = subject_indices[n_train + n_val :]

        if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
            yield subject_id, train_idx, val_idx, test_idx

def run_loso_experiment(df, feature_columns, config):
    fold_rows = []
    prediction_rows = []
    global_start = perf_counter()
    subjects = list(iter_loso_splits(df))

    for fold_index, (subject_id, train_idx, test_idx) in enumerate(subjects, start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_columns].to_numpy(dtype=np.float64)
        y_train = train_df["target"].to_numpy(dtype=int)
        X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
        y_test = test_df["target"].to_numpy(dtype=int)

        if np.unique(y_train).size < 2:
            raise ValueError(f"Training fold for subject {subject_id} has only one class.")

        fold_start = perf_counter()
        adaptation = get_adaptation(config.adaptation_name)
        X_train_adapted, X_test_adapted = adaptation.fit_transform(X_train, X_test, y_train=y_train)

        model = get_model(config.model_name, seed=config.seed)
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError(
                f"Model '{config.model_name}' is not a tabular sklearn-style model. "
                "Use this LOSO runner for feature-based baselines, or add a deep LOSO "
                "trainer that loads epoched EEG tensors."
            )
        model.fit(X_train_adapted, y_train)
        y_pred = model.predict(X_test_adapted)
        y_score = try_predict_scores(model, X_test_adapted)

        log_sklearn_model(model, X_train, artifact_path=f"models_fold_{fold_index:02d}")

        metrics = binary_metrics(y_test, y_pred)
        metrics["roc_auc"] = safe_roc_auc(y_test, y_score)
        metrics.update(confusion_counts(y_test, y_pred))
        metrics.update(
            {
                "fold": fold_index,
                "subject_id": subject_id,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_alert": int((y_train == 0).sum()),
                "n_train_drowsy": int((y_train == 1).sum()),
                "n_test_alert": int((y_test == 0).sum()),
                "n_test_drowsy": int((y_test == 1).sum()),
                "fold_seconds": perf_counter() - fold_start,
                "adaptation_uses_target_unlabeled": bool(getattr(adaptation, "uses_target_data", False)),
            }
        )
        fold_rows.append(metrics)

        fold_predictions = test_df[["subject_id", "recording", "epoch_index", "perclos", "target"]].copy()
        fold_predictions["fold"] = fold_index
        fold_predictions["y_pred"] = y_pred
        fold_predictions["y_score"] = y_score if y_score is not None else np.nan
        prediction_rows.append(fold_predictions)

        print(
            f"[{fold_index:02d}/{len(subjects):02d}] subject={subject_id} "
            f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
            f"f1={metrics['f1']:.3f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    print(f"Finished LOSO in {perf_counter() - global_start:.1f}s")
    return fold_metrics, predictions
