import sys
from pathlib import Path

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
    log_fold_metrics,
    log_summary_metrics,
)
from src.utils.config import parse_experiment_args
from src.utils.seed import set_global_seed


def build_run_name(config):
    if config.run_name:
        return config.run_name
    return f"{config.dataset_name}_{config.model_name}_{config.adaptation_name}_seed{config.seed}"


def run_experiment(config):
    set_global_seed(config.seed)
    configure_mlflow(config)

    df, feature_columns, dropped_columns = load_labeled_features(
        config.feature_path,
        alert_threshold=config.alert_threshold,
        drowsy_threshold=config.drowsy_threshold,
    )

    run_name = build_run_name(config)
    output_dir = Path(config.output_dir) / run_name

    with mlflow.start_run(run_name=run_name):
        log_config(config)

        fold_metrics, predictions = run_loso_experiment(df, feature_columns, config)
        summary = aggregate_fold_metrics(fold_metrics)
        artifacts = collect_artifacts(
            output_dir,
            df,
            fold_metrics,
            predictions,
            summary,
            save_plots=config.save_plots,
        )

        log_dataset_metadata(config, df, feature_columns, fold_metrics, dropped_columns)
        log_fold_metrics(fold_metrics)
        log_summary_metrics(summary)
        log_artifacts(artifacts)

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
