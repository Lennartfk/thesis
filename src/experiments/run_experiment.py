import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.data.load_data import load_labeled_features
from src.experiments.evaluate import aggregate_fold_metrics
from src.experiments.loso import run_loso_experiment
from src.tracking.artifacts import collect_artifacts
from src.tracking.mlflow_utils import (
    configure_mlflow,
    log_artifacts,
    log_config,
    log_dataset_metadata,
    log_epoch_history,
    log_fold_metrics,
    register_logged_model,
    log_summary_metrics,
)
from src.utils.config import parse_experiment_args
from src.utils.seed import set_global_seed


def build_run_name(config):
    if config.run_name:
        return config.run_name
    return f"{config.dataset_name}_{config.model_name}_{config.adaptation_name}_seed{config.seed}"


def build_registry_model_name(config):
    if config.registry_model_name:
        return config.registry_model_name

    auto_name = f"{config.dataset_name}_{config.model_family}_{config.model_name}_{config.adaptation_name}"
    auto_name = auto_name.lower().replace("-", "_")
    auto_name = re.sub(r"[^a-z0-9_.]", "_", auto_name)
    auto_name = re.sub(r"_+", "_", auto_name).strip("_")
    return auto_name


def resolve_output_dir(base_output_dir, run_name):
    base_path = Path(base_output_dir) / run_name
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=False)
        return base_path

    suffix = 2
    while True:
        candidate_path = Path(base_output_dir) / f"{run_name}_{suffix}"
        if not candidate_path.exists():
            candidate_path.mkdir(parents=True, exist_ok=False)
            return candidate_path
        suffix += 1


def run_experiment(config):
    set_global_seed(config.seed)
    configure_mlflow(config)

    run_name = build_run_name(config)
    output_dir = resolve_output_dir(config.output_dir, run_name)

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        log_config(config)

        if config.model_family == "sklearn":
            df, feature_columns, dropped_columns = load_labeled_features(
                config.feature_path,
                alert_threshold=config.alert_threshold,
                drowsy_threshold=config.drowsy_threshold,
            )
            fold_metrics, predictions = run_loso_experiment(df, feature_columns, config)
            history = None
            feature_count = len(feature_columns)
            extra_artifacts = []
        elif config.model_family == "deep":
            from src.data.eeg_dataset import build_epoch_index
            from src.experiments.deep_loso import run_deep_loso_experiment

            df = build_epoch_index(
                config.epoch_dir,
                alert_threshold=config.alert_threshold,
                drowsy_threshold=config.drowsy_threshold,
            )
            feature_columns = []
            dropped_columns = []
            fold_metrics, predictions, history, deep_metadata = run_deep_loso_experiment(df, config, output_dir)
            feature_count = deep_metadata["feature_count"]
            extra_artifacts = deep_metadata["checkpoint_paths"]
            mlflow.log_params(
                {
                    "n_channels": deep_metadata["n_channels"],
                    "n_samples": deep_metadata["n_samples"],
                    "sfreq": deep_metadata["sfreq"],
                    "channels": deep_metadata["channels"],
                }
            )
        else:
            raise ValueError(f"Unknown model_family '{config.model_family}'. Use 'sklearn' or 'deep'.")

        summary = aggregate_fold_metrics(fold_metrics)
        artifacts = collect_artifacts(
            output_dir,
            df,
            fold_metrics,
            predictions,
            summary,
            save_plots=config.save_plots,
            history=history,
        )
        artifacts.extend(extra_artifacts)

        log_dataset_metadata(config, df, feature_columns, fold_metrics, dropped_columns, feature_count=feature_count)
        log_fold_metrics(fold_metrics)
        log_epoch_history(history)
        log_summary_metrics(summary)
        log_artifacts(artifacts)

        if config.register_model and not fold_metrics.empty:
            ranking_metric = "balanced_accuracy"
            if ranking_metric not in fold_metrics.columns:
                ranking_metric = "accuracy"
            best_row = fold_metrics.sort_values(ranking_metric, ascending=False).iloc[0]
            best_fold = int(best_row["fold"])

            registry_name = build_registry_model_name(config)

            model_version = register_logged_model(
                run_id=run_id,
                artifact_path=f"models_fold_{best_fold:02d}",
                model_name=registry_name,
            )
            print(
                f"Registered model '{registry_name}' as version {model_version.version} "
                f"from fold {best_fold} ({ranking_metric}={best_row[ranking_metric]:.3f})."
            )

    print("")
    print("Aggregate metrics")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved experiment artifacts to {output_dir}")
    return fold_metrics, predictions, summary


def main():
    config = parse_experiment_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
