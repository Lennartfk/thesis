from dataclasses import asdict
from pathlib import Path

import mlflow


def configure_mlflow(config):
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


def log_config(config):
    for key, value in asdict(config).items():
        mlflow.log_param(key, value)


def log_dataset_metadata(config, df, feature_columns, fold_metrics, dropped_columns):
    mlflow.log_params(
        {
            "fold_count": int(fold_metrics["fold"].nunique()),
            "feature_count": len(feature_columns),
            "sample_count": len(df),
            "subject_count": int(df["subject_id"].nunique()),
            "dropped_non_numeric_feature_columns": ",".join(dropped_columns),
        }
    )
    mlflow.log_metric("total_train_samples_across_folds", int(fold_metrics["n_train"].sum()))
    mlflow.log_metric("total_test_samples_across_folds", int(fold_metrics["n_test"].sum()))


def log_fold_metrics(fold_metrics):
    for _, row in fold_metrics.iterrows():
        step = int(row["fold"])
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
            value = row.get(metric)
            if value == value:
                mlflow.log_metric(f"fold_{metric}", float(value), step=step)


def log_summary_metrics(summary):
    for key, value in summary.iloc[0].items():
        if value == value:
            mlflow.log_metric(key, float(value))


def log_artifacts(paths):
    for path in paths:
        mlflow.log_artifact(str(Path(path)))
