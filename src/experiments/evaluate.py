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
