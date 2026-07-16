#!/usr/bin/env python3
"""
Quick script to add files/directories to MLflow experiment artifacts.

Usage:
    python scripts/add_artifacts.py <experiment_name> <run_id> <file_or_dir_path> [additional_paths...]
    python scripts/add_artifacts.py --list-experiments
    python scripts/add_artifacts.py --list-runs <experiment_name>
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.paths import MLRUNS_DIR


def configure_mlflow():
    """Set MLflow tracking URI to SQLite database."""
    db_uri = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(db_uri)


def list_experiments():
    """List all available experiments."""
    configure_mlflow()
    experiments = mlflow.search_experiments()
    print("\nAvailable Experiments:")
    print("-" * 60)
    for exp in experiments:
        print(f"  Name: {exp.name:40} | ID: {exp.experiment_id}")
    print()


def list_runs(experiment_name):
    """List all runs in an experiment."""
    configure_mlflow()
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found.")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"\nRuns in experiment '{experiment_name}':")
        print("-" * 80)
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_name = run.get("tags.mlflow.runName", "N/A")
            status = run["status"]
            print(f"  Run ID: {run_id:8} | Name: {run_name:30} | Status: {status}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")


def add_artifacts(experiment_name, run_id, file_paths):
    """Add files/directories to run artifacts."""
    configure_mlflow()

    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found.")
            return False

        # Set the experiment and use the run
        mlflow.set_experiment(experiment_name)

        # Log each artifact
        for file_path_str in file_paths:
            file_path = Path(file_path_str)

            if not file_path.exists():
                print(f"⚠️  Path does not exist: {file_path}")
                continue

            with mlflow.start_run(run_id=run_id, nested=False) as run:
                if file_path.is_file():
                    # Log single file
                    mlflow.log_artifact(str(file_path))
                    print(f"✅ Added file: {file_path.name}")
                elif file_path.is_dir():
                    # Log directory
                    mlflow.log_artifacts(str(file_path))
                    print(f"✅ Added directory: {file_path.name}")

        print(f"\n✅ Artifacts successfully added to run {run_id}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def delete_artifacts(experiment_name, run_id, artifact_paths):
    """Delete artifact paths from a run."""
    configure_mlflow()

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found.")
            return False

        client = MlflowClient()
        for artifact_path in artifact_paths:
            client.delete_artifacts(run_id, artifact_path)
            print(f"✅ Deleted artifact path: {artifact_path}")

        print(f"\n✅ Artifact deletion completed for run {run_id}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add files to MLflow experiment artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python scripts/add_artifacts.py --list-experiments

  # List runs in an experiment
  python scripts/add_artifacts.py --list-runs seedvig_loso

  # Add a file to a run
  python scripts/add_artifacts.py seedvig_loso abc123def456 ./my_file.csv

  # Add multiple files
  python scripts/add_artifacts.py seedvig_loso abc123def456 ./file1.csv ./file2.png ./output_dir/

    # Delete artifact path(s) from a run
    python scripts/add_artifacts.py seedvig_loso abc123def456 --delete-artifact confusion_matrix.png
    python scripts/add_artifacts.py seedvig_loso abc123def456 --delete-artifact plots --delete-artifact tables/fold_metrics.csv
        """,
    )

    parser.add_argument("--list-experiments", action="store_true", help="List all experiments")
    parser.add_argument("--list-runs", metavar="EXPERIMENT_NAME", help="List runs in an experiment")
    parser.add_argument(
        "--delete-artifact",
        action="append",
        metavar="ARTIFACT_PATH",
        help="Delete artifact path from a run (can be passed multiple times)",
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name")
    parser.add_argument("run_id", nargs="?", help="Run ID")
    parser.add_argument("files", nargs="*", help="File or directory paths to add as artifacts")

    args = parser.parse_args()

    if args.list_experiments:
        list_experiments()
    elif args.list_runs:
        list_runs(args.list_runs)
    elif args.experiment and args.run_id and args.delete_artifact:
        delete_artifacts(args.experiment, args.run_id, args.delete_artifact)
    elif args.experiment and args.run_id and args.files:
        add_artifacts(args.experiment, args.run_id, args.files)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
