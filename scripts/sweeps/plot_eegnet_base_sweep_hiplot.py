"""Generate an interactive HiPlot HTML for the EEGNet base sweep.

This script reads completed sweep runs under data/results/sweeps/eegnet_base
and the trial config files under configs/sweeps/eegnet_base, creates a tidy
DataFrame of hyperparameters + metric, and exports a HiPlot HTML experiment.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "data/results/sweeps/eegnet_base"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs/sweeps/eegnet_base"
DEFAULT_OUTPUT = DEFAULT_SWEEP_DIR / "eegnet_base_hiplot.html"


def extract_trial_index(run_name):
    match = re.search(r"_t(\d{3})_", run_name)
    if not match:
        raise ValueError(f"Could not extract trial index from run name: {run_name}")
    return match.group(1)


def load_trial_config(config_dir, run_name):
    trial_index = extract_trial_index(run_name)
    config_path = config_dir / f"trial_{trial_index}.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_completed_runs(sweep_dir, config_dir):
    rows = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary_metrics.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        config = load_trial_config(config_dir, run_dir.name)
        row = {"run_name": run_dir.name}
        if "balanced_accuracy_mean" in summary.columns:
            row["balanced_accuracy_mean"] = float(summary.loc[0, "balanced_accuracy_mean"])
        elif "balanced_accuracy" in summary.columns:
            row["balanced_accuracy_mean"] = float(summary.loc[0, "balanced_accuracy"])
        else:
            row["balanced_accuracy_mean"] = None

        for key in [
            "learning_rate",
            "batch_size",
            "weight_decay",
            "eegnet_dropout",
            "eegnet_f1",
            "eegnet_depth_multiplier",
            "eegnet_temporal_kernel_length",
            "eegnet_separable_kernel_length",
            "tune_decision_threshold",
        ]:
            row[key] = config.get(key)

        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed runs found under {sweep_dir}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export HiPlot HTML for EEGNet sweep")
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    config_dir = Path(args.config_dir)
    output = Path(args.output)

    df = load_completed_runs(sweep_dir, config_dir)

    df["learning_rate"] = df["learning_rate"].astype(float)
    df["batch_size"] = df["batch_size"].astype(int)
    df["weight_decay"] = df["weight_decay"].astype(float)
    df["eegnet_dropout"] = df["eegnet_dropout"].astype(float)
    df["eegnet_f1"] = df["eegnet_f1"].astype(int)
    df["eegnet_depth_multiplier"] = df["eegnet_depth_multiplier"].astype(int)
    df["eegnet_temporal_kernel_length"] = df["eegnet_temporal_kernel_length"].astype(int)
    df["eegnet_separable_kernel_length"] = df["eegnet_separable_kernel_length"].astype(int)
    df["tune_decision_threshold"] = df["tune_decision_threshold"].astype(bool)

    try:
        import hiplot as hp
    except Exception as exc:
        raise SystemExit("HiPlot is required. Install with: pip install hiplot") from exc

    exp = hp.Experiment.from_dataframe(df)
    exp.to_html(str(output), open_in_browser=False)
    print(f"Saved HiPlot HTML to {output}")


if __name__ == "__main__":
    main()
