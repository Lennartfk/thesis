from dataclasses import asdict
from pathlib import Path
import re

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import torch


_INVALID_LOGGED_MODEL_CHARS = r"[/:.%\"']"


def _logged_model_name(name_or_path):
    sanitized = re.sub(_INVALID_LOGGED_MODEL_CHARS, "_", str(name_or_path)).strip("_")
    if not sanitized:
        raise ValueError("Model name cannot be empty after sanitization.")
    return sanitized


def configure_mlflow(config):
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


def log_config(config):
    for key, value in asdict(config).items():
        mlflow.log_param(key, value)


def log_dataset_metadata(config, df, feature_columns, fold_metrics, dropped_columns, feature_count=None):
    if feature_count is None:
        feature_count = len(feature_columns)

    mlflow.log_params(
        {
            "fold_count": int(fold_metrics["fold"].nunique()),
            "feature_count": feature_count,
            "sample_count": len(df),
            "subject_count": int(df["subject_id"].nunique()),
            "dropped_non_numeric_feature_columns": ",".join(dropped_columns),
        }
    )
    mlflow.log_metric("total_train_samples_across_folds", int(fold_metrics["n_train"].sum()))
    mlflow.log_metric("total_test_samples_across_folds", int(fold_metrics["n_test"].sum()))


def log_subject_balance_metadata(config, raw_df, filtered_df, subject_balance, excluded_subjects):
    excluded_ids = sorted(excluded_subjects["subject_id"].astype(str).tolist())
    retained_ids = sorted(subject_balance.loc[subject_balance["retained"], "subject_id"].astype(str).tolist())

    mlflow.log_params(
        {
            "exclude_imbalanced_subjects_enabled": bool(config.exclude_imbalanced_subjects),
            "raw_subject_count": int(raw_df["subject_id"].nunique()),
            "retained_subject_count": int(filtered_df["subject_id"].nunique()),
            "excluded_subject_count": len(excluded_ids),
            "excluded_subject_ids": ",".join(excluded_ids),
            "retained_subject_ids": ",".join(retained_ids),
        }
    )
    mlflow.log_metric("raw_sample_count", int(len(raw_df)))
    mlflow.log_metric("retained_sample_count", int(len(filtered_df)))
    mlflow.log_metric("excluded_sample_count", int(len(raw_df) - len(filtered_df)))

    if "target" in raw_df.columns:
        excluded_df = raw_df.loc[raw_df["subject_id"].astype(str).isin(excluded_ids)]
        mlflow.log_metric("excluded_alert_count", int((excluded_df["target"] == 0).sum()))
        mlflow.log_metric("excluded_drowsy_count", int((excluded_df["target"] == 1).sum()))
        mlflow.log_metric("retained_alert_count", int((filtered_df["target"] == 0).sum()))
        mlflow.log_metric("retained_drowsy_count", int((filtered_df["target"] == 1).sum()))


def log_fold_metrics(fold_metrics):
    # Log per-fold test metrics with explicit fold numbering and metric type
    for _, row in fold_metrics.iterrows():
        fold = int(row["fold"])
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
            value = row.get(metric)
            if value == value:
                mlflow.log_metric(f"fold_{fold:02d}_test_{metric}", float(value), step=fold)


def log_summary_metrics(summary):
    for key, value in summary.iloc[0].items():
        if value == value:
            mlflow.log_metric(key, float(value))


def log_epoch_history(history):
    if history is None or history.empty:
        return

    for _, row in history.iterrows():
        step = int(row["epoch"])
        fold = int(row["fold"])
        for key, value in row.items():
            if key in {"fold", "epoch", "subject_id", "test_subject_id", "val_subject_id"}:
                continue
            if value == value:
                mlflow.log_metric(f"fold_{fold:02d}_{key}", float(value), step=step)


def log_artifacts(paths):
    for path in paths:
        mlflow.log_artifact(str(Path(path)))


def log_sklearn_model(model, X_example, artifact_path):
    signature = infer_signature(X_example, model.predict(X_example)) if X_example is not None else None
    model_name = _logged_model_name(artifact_path)
    try:
        mlflow.sklearn.log_model(
            model,
            name=model_name,
            signature=signature,
            input_example=X_example,
        )
    except TypeError:
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            signature=signature,
            input_example=X_example,
        )


def log_torch_model(model, input_example, artifact_path):
    if isinstance(input_example, torch.Tensor):
        input_example = input_example.detach().cpu()

    model_cpu = model.detach() if hasattr(model, "detach") else model
    if hasattr(model_cpu, "cpu"):
        model_cpu = model_cpu.cpu()
    if hasattr(model_cpu, "eval"):
        model_cpu.eval()

    model_name = _logged_model_name(artifact_path)
    try:
        mlflow.pytorch.log_model(
            model_cpu,
            name=model_name,
            input_example=input_example,
        )
    except TypeError:
        mlflow.pytorch.log_model(
            model_cpu,
            artifact_path=model_name,
            input_example=input_example,
        )


def register_logged_model(run_id, artifact_path, model_name):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.register_model(model_uri=model_uri, name=model_name)
