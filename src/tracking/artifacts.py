from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def save_tables(output_dir, fold_metrics, predictions, summary):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold_metrics": output_path / "fold_metrics.csv",
        "predictions": output_path / "predictions.csv",
        "summary": output_path / "summary_metrics.csv",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def save_confusion_matrix(output_dir, predictions):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "confusion_matrix.png"

    matrix = confusion_matrix(predictions["target"], predictions["y_pred"], labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["alert", "drowsy"])
    display.plot(values_format="d", colorbar=False)
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    return artifact_path


def save_roc_curve(output_dir, predictions):
    if predictions["target"].nunique() < 2 or predictions["y_score"].isna().all():
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "roc_curve.png"

    usable = predictions.dropna(subset=["y_score"])
    RocCurveDisplay.from_predictions(usable["target"], usable["y_score"])
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    return artifact_path


def save_class_balance(output_dir, df):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "class_balance_by_subject.csv"

    balance = (
        df.groupby("subject_id")["target"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "alert", 1: "drowsy"})
        .reset_index()
    )
    balance.to_csv(artifact_path, index=False)
    return artifact_path


def collect_artifacts(output_dir, df, fold_metrics, predictions, summary, save_plots=True):
    paths = list(save_tables(output_dir, fold_metrics, predictions, summary).values())
    paths.append(save_class_balance(output_dir, df))

    if save_plots and not predictions.empty:
        paths.append(save_confusion_matrix(output_dir, predictions))
        roc_path = save_roc_curve(output_dir, predictions)
        if roc_path is not None:
            paths.append(roc_path)

    return [Path(path) for path in paths if path is not None]
