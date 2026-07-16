import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize_run(run_dir):
    fold_metrics_path = run_dir / "fold_metrics.csv"
    if not fold_metrics_path.exists():
        return None

    fold_metrics = pd.read_csv(fold_metrics_path)
    if fold_metrics.empty:
        return None

    row = {
        "run_name": run_dir.name,
        "folds": int(fold_metrics["fold"].nunique()),
        "val_metric": fold_metrics["best_val_metric_name"].mode().iloc[0],
        "val_metric_mean": fold_metrics["best_val_metric"].mean(),
        "val_metric_std": fold_metrics["best_val_metric"].std(),
    }
    for metric in ["balanced_accuracy", "f1", "roc_auc", "accuracy"]:
        if metric in fold_metrics:
            row[f"test_{metric}_mean"] = fold_metrics[metric].mean()
            row[f"test_{metric}_std"] = fold_metrics[metric].std()

    config_path = run_dir / "config.json"
    if config_path.exists():
        row["config_path"] = str(config_path.relative_to(PROJECT_ROOT))
    return row


def main():
    parser = argparse.ArgumentParser(description="Rank completed EEGNet sweep trials.")
    parser.add_argument("--sweep-dir", default="data/results/sweeps/eegnet_base")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    global pd
    import pandas as pd

    sweep_dir = PROJECT_ROOT / args.sweep_dir
    rows = []
    for run_dir in sorted(path for path in sweep_dir.glob("*") if path.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No completed sweep runs found under {sweep_dir}")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["val_metric_mean", "test_f1_mean", "test_balanced_accuracy_mean"],
        ascending=[False, False, False],
    )
    print(summary.head(args.top).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
