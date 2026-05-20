from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def _subject_sort_key(subject_id):
    subject_text = str(subject_id)
    try:
        return (0, int(subject_text))
    except ValueError:
        return (1, subject_text)


def save_tables(output_dir, fold_metrics, predictions, summary):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold_metrics": output_path / "fold_metrics.csv",
        "predictions": output_path / "predictions.csv",
        "summary": output_path / "summary_metrics.csv",
        "fold_summary": output_path / "fold_summary.csv",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    summary.to_csv(paths["summary"], index=False)

    # Save detailed per-fold summary table if train/val/test columns are present
    if not fold_metrics.empty and "test_subject_id" in fold_metrics.columns:
        fold_summary = fold_metrics.loc[
            :, [
                "fold",
                "train_subjects",
                "val_subject_id",
                "test_subject_id",
                "accuracy",
                "balanced_accuracy",
                "f1",
            ]
        ].rename(columns={"val_subject_id": "val_subject", "test_subject_id": "test_subject"})
        fold_summary.to_csv(paths["fold_summary"], index=False)
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


def save_relative_confusion_matrix(output_dir, predictions):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "confusion_matrix_relative.png"

    matrix = confusion_matrix(
        predictions["target"],
        predictions["y_pred"],
        labels=[0, 1],
        normalize="true",
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=["alert", "drowsy"])
    display.plot(values_format=".2f", colorbar=False)
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


def _save_test_metric_bar(output_dir, fold_metrics, metric, filename):
    if fold_metrics.empty or metric not in fold_metrics.columns:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / filename

    # Use test_subject_id for x-axis when available, otherwise fallback to fold
    if "test_subject_id" in fold_metrics.columns:
        sorted_metrics = fold_metrics.copy()
        sorted_metrics["subject_sort"] = sorted_metrics["test_subject_id"].map(_subject_sort_key)
        sorted_metrics = sorted_metrics.sort_values("subject_sort").reset_index(drop=True)
        x = sorted_metrics["test_subject_id"].astype(str)
        y = sorted_metrics[metric]
    else:
        sorted_metrics = fold_metrics.sort_values("fold").reset_index(drop=True)
        x = sorted_metrics["fold"].astype(str)
        y = sorted_metrics[metric]

    mean = float(y.mean())
    std = float(y.std())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color="tab:blue")
    ax.set_xlabel("test subject")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} by test subject — mean={mean:.3f} ± std={std:.3f}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=150)
    plt.close(fig)
    return artifact_path


def save_training_history(output_dir, history):
    if history is None or history.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "training_history.csv"
    history.to_csv(artifact_path, index=False)
    return artifact_path


def save_learning_curves(output_dir, history):
    if history is None or history.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "learning_curves.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (_, fold_history) in enumerate(history.groupby("fold")):
        show_label = idx == 0
        axes[0].plot(
            fold_history["epoch"],
            fold_history["train_loss"],
            alpha=0.35,
            color="tab:blue",
            label="train loss" if show_label else None,
        )
        axes[0].plot(
            fold_history["epoch"],
            fold_history["val_loss"],
            alpha=0.35,
            color="tab:orange",
            label="validation loss" if show_label else None,
        )
        axes[1].plot(
            fold_history["epoch"],
            fold_history["train_balanced_accuracy"],
            alpha=0.25,
            color="tab:blue",
            linestyle="--",
            label="train balanced accuracy" if show_label else None,
        )
        axes[1].plot(
            fold_history["epoch"],
            fold_history["val_balanced_accuracy"],
            alpha=0.35,
            color="tab:green",
            label="validation balanced accuracy" if show_label else None,
        )
        axes[2].plot(
            fold_history["epoch"],
            fold_history["train_accuracy"],
            alpha=0.25,
            color="tab:blue",
            linestyle="--",
            label="train accuracy" if show_label else None,
        )
        axes[2].plot(
            fold_history["epoch"],
            fold_history["val_accuracy"],
            alpha=0.35,
            color="tab:green",
            label="validation accuracy" if show_label else None,
        )

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best", frameon=False)

    axes[1].set_title("Balanced Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].legend(loc="best", frameon=False)

    axes[2].set_title("Accuracy Curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=150)
    plt.close(fig)
    return artifact_path


def save_per_fold_learning_curves(output_dir, history):
    if history is None or history.empty:
        return []

    output_path = Path(output_dir) / "learning_curves_by_fold"
    output_path.mkdir(parents=True, exist_ok=True)

    paths = []
    for fold, fold_history in history.groupby("fold"):
        fold_history = fold_history.sort_values("epoch")
        test_subject = str(fold_history["test_subject_id"].iloc[0])
        val_subject = str(fold_history["val_subject_id"].iloc[0])

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        fig.suptitle(f"Fold {int(fold):02d}: test subject {test_subject}, validation subject {val_subject}")

        axes[0, 0].plot(fold_history["epoch"], fold_history["train_loss"], label="train", color="tab:blue")
        axes[0, 0].plot(fold_history["epoch"], fold_history["val_loss"], label="validation", color="tab:orange")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_ylabel("Cross-Entropy Loss")
        axes[0, 0].legend(frameon=False)

        axes[0, 1].plot(fold_history["epoch"], fold_history["train_balanced_accuracy"], label="train", color="tab:blue")
        axes[0, 1].plot(fold_history["epoch"], fold_history["val_balanced_accuracy"], label="validation", color="tab:green")
        axes[0, 1].set_title("Balanced Accuracy")
        axes[0, 1].set_ylabel("Balanced Accuracy")
        axes[0, 1].legend(frameon=False)

        axes[1, 0].plot(fold_history["epoch"], fold_history["train_accuracy"], label="train", color="tab:blue")
        axes[1, 0].plot(fold_history["epoch"], fold_history["val_accuracy"], label="validation", color="tab:green")
        axes[1, 0].set_title("Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend(frameon=False)

        axes[1, 1].plot(fold_history["epoch"], fold_history["train_f1"], label="train", color="tab:blue")
        axes[1, 1].plot(fold_history["epoch"], fold_history["val_f1"], label="validation", color="tab:green")
        axes[1, 1].set_title("F1")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1")
        axes[1, 1].legend(frameon=False)

        for ax in axes.ravel():
            ax.grid(alpha=0.2)

        artifact_path = output_path / f"fold_{int(fold):02d}_test_{test_subject}_val_{val_subject}_learning_curves.png"
        plt.tight_layout()
        plt.savefig(artifact_path, dpi=150)
        plt.close(fig)
        paths.append(artifact_path)

    return paths


def save_validation_test_metric_scatter(output_dir, fold_metrics, history):
    if history is None or history.empty or fold_metrics.empty:
        return None
    if "best_epoch" not in fold_metrics.columns or "balanced_accuracy" not in fold_metrics.columns:
        return None
    subject_column = "test_subject_id" if "test_subject_id" in fold_metrics.columns else "subject_id"
    if subject_column not in fold_metrics.columns:
        return None

    rows = []
    for _, fold_row in fold_metrics.iterrows():
        fold = int(fold_row["fold"])
        best_epoch = int(fold_row["best_epoch"])
        match = history[(history["fold"] == fold) & (history["epoch"] == best_epoch)]
        if match.empty:
            continue
        history_row = match.iloc[0]
        rows.append(
            {
                "test_subject_id": str(fold_row[subject_column]),
                "val_balanced_accuracy": float(history_row["val_balanced_accuracy"]),
                "test_balanced_accuracy": float(fold_row["balanced_accuracy"]),
            }
        )

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    frame["subject_sort"] = frame["test_subject_id"].map(_subject_sort_key)
    frame = frame.sort_values("subject_sort")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "validation_vs_test_balanced_accuracy.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame["val_balanced_accuracy"], frame["test_balanced_accuracy"], color="tab:blue")
    for _, row in frame.iterrows():
        ax.annotate(row["test_subject_id"], (row["val_balanced_accuracy"], row["test_balanced_accuracy"]), fontsize=8)
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Validation balanced accuracy at selected epoch")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_title("Validation vs Test Performance by Fold")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=150)
    plt.close(fig)
    return artifact_path


def save_prediction_timelines(output_dir, predictions):
    if predictions.empty or "y_score" not in predictions.columns or predictions["y_score"].isna().all():
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "prediction_timelines_by_subject.png"

    subjects = sorted(predictions["subject_id"].astype(str).unique(), key=_subject_sort_key)
    n_cols = 3
    n_rows = (len(subjects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3, 2.4 * n_rows)), sharey=True)
    axes = axes.ravel()

    for ax, subject_id in zip(axes, subjects):
        subject_predictions = predictions[predictions["subject_id"].astype(str) == subject_id].sort_values("epoch_index")
        ax.plot(
            subject_predictions["epoch_index"],
            subject_predictions["perclos"],
            color="tab:orange",
            linewidth=1.2,
            label="PERCLOS",
        )
        ax.plot(
            subject_predictions["epoch_index"],
            subject_predictions["y_score"],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.85,
            label="drowsy score",
        )
        ax.axhline(0.35, color="tab:green", linestyle="--", linewidth=0.8)
        ax.axhline(0.70, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(f"Subject {subject_id}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.15)

    for ax in axes[len(subjects):]:
        ax.axis("off")

    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("PERCLOS and Model Drowsy Score Across Held-Out Test Epochs", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(artifact_path, dpi=150)
    plt.close(fig)
    return artifact_path


def save_binary_prediction_timelines(output_dir, predictions):
    if predictions.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / "binary_prediction_timelines_by_subject.png"

    subjects = sorted(predictions["subject_id"].astype(str).unique(), key=_subject_sort_key)
    n_cols = 3
    n_rows = (len(subjects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3, 2.3 * n_rows)), sharey=True)
    axes = axes.ravel()

    for ax, subject_id in zip(axes, subjects):
        subject_predictions = predictions[predictions["subject_id"].astype(str) == subject_id].sort_values("epoch_index")
        ax.step(
            subject_predictions["epoch_index"],
            subject_predictions["target"],
            where="mid",
            color="tab:orange",
            linewidth=1.3,
            label="true target",
        )
        ax.step(
            subject_predictions["epoch_index"],
            subject_predictions["y_pred"],
            where="mid",
            color="tab:blue",
            linewidth=1.0,
            alpha=0.85,
            label="predicted target",
        )
        ax.set_title(f"Subject {subject_id}")
        ax.set_xlabel("Epoch")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["alert", "drowsy"])
        ax.set_ylim(-0.15, 1.15)
        ax.grid(alpha=0.15)

    for ax in axes[len(subjects):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Binarized PERCLOS Target and Model Prediction Across Held-Out Test Epochs", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(artifact_path, dpi=150)
    plt.close(fig)
    return artifact_path


def collect_artifacts(output_dir, df, fold_metrics, predictions, summary, save_plots=True, history=None):
    paths = list(save_tables(output_dir, fold_metrics, predictions, summary).values())
    paths.append(save_class_balance(output_dir, df))
    paths.append(save_training_history(output_dir, history))

    if save_plots and not predictions.empty:
        paths.append(save_confusion_matrix(output_dir, predictions))
        paths.append(save_relative_confusion_matrix(output_dir, predictions))
        roc_path = save_roc_curve(output_dir, predictions)
        if roc_path is not None:
            paths.append(roc_path)
        learning_curve_path = save_learning_curves(output_dir, history)
        if learning_curve_path is not None:
            paths.append(learning_curve_path)
        paths.extend(save_per_fold_learning_curves(output_dir, history))
        validation_test_path = save_validation_test_metric_scatter(output_dir, fold_metrics, history)
        if validation_test_path is not None:
            paths.append(validation_test_path)
        prediction_timeline_path = save_prediction_timelines(output_dir, predictions)
        if prediction_timeline_path is not None:
            paths.append(prediction_timeline_path)
        binary_prediction_timeline_path = save_binary_prediction_timelines(output_dir, predictions)
        if binary_prediction_timeline_path is not None:
            paths.append(binary_prediction_timeline_path)

    # Save test metric bar plots (accuracy, balanced_accuracy, f1)
    paths.append(_save_test_metric_bar(output_dir, fold_metrics, "accuracy", "test_accuracy_by_subject.png"))
    paths.append(_save_test_metric_bar(output_dir, fold_metrics, "balanced_accuracy", "test_balanced_accuracy_by_subject.png"))
    paths.append(_save_test_metric_bar(output_dir, fold_metrics, "f1", "test_f1_by_subject.png"))

    return [Path(path) for path in paths if path is not None]
