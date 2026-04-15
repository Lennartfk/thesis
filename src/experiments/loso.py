from time import perf_counter

import numpy as np
import pandas as pd

from src.experiments.evaluate import binary_metrics, confusion_counts, safe_roc_auc, try_predict_scores
from src.experiments.registry import get_adaptation, get_model


def iter_loso_splits(df, subject_column="subject_id"):
    for subject_id in sorted(df[subject_column].astype(str).unique()):
        test_mask = df[subject_column].astype(str) == subject_id
        yield subject_id, np.flatnonzero(~test_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())


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
        model.fit(X_train_adapted, y_train)
        y_pred = model.predict(X_test_adapted)
        y_score = try_predict_scores(model, X_test_adapted)

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
