import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def balanced_accuracy_present_classes(y_true, y_pred, labels=(0, 1)):
    recalls = []
    for label in labels:
        label_mask = y_true == label
        if np.any(label_mask):
            recalls.append(float(np.mean(y_pred[label_mask] == label)))
    return float(np.mean(recalls)) if recalls else np.nan


def binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_present_classes(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def binary_predictions_from_scores(y_score, threshold=0.5):
    return (np.asarray(y_score) >= float(threshold)).astype(int)


def select_binary_threshold(y_true, y_score, metric="balanced_accuracy", default_threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold), np.nan, {}

    unique_scores = np.unique(y_score)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2 if unique_scores.size > 1 else np.array([])
    candidates = np.unique(np.concatenate(([0.0, float(default_threshold), 1.0], unique_scores, midpoints)))

    best_threshold = float(default_threshold)
    best_value = None
    best_metrics = {}
    for threshold in candidates:
        y_pred = binary_predictions_from_scores(y_score, threshold)
        metrics = binary_metrics(y_true, y_pred)
        if metric not in metrics:
            raise KeyError(f"Unknown threshold metric '{metric}'. Available: {list(metrics.keys())}")

        value = metrics[metric]
        is_better = best_value is None or value > best_value
        is_tie = best_value is not None and np.isclose(value, best_value)
        closer_to_default = abs(float(threshold) - default_threshold) < abs(best_threshold - default_threshold)
        if is_better or (is_tie and closer_to_default):
            best_threshold = float(threshold)
            best_value = float(value)
            best_metrics = metrics

    return best_threshold, float(best_value), best_metrics


def try_predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def safe_roc_auc(y_true, y_score):
    if y_score is None or np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def confusion_counts(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def aggregate_fold_metrics(fold_metrics):
    summary = {}
    if fold_metrics.empty:
        return pd.DataFrame()

    for metric in METRIC_COLUMNS + ["roc_auc"]:
        if metric in fold_metrics.columns:
            summary[f"{metric}_mean"] = fold_metrics[metric].mean()
            summary[f"{metric}_std"] = fold_metrics[metric].std()
    return pd.DataFrame([summary])
