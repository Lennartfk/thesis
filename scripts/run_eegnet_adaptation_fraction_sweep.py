import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.run_experiment import run_experiment
from src.utils.config import load_config


DEFAULT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_METHODS = ["ea", "adabn"]


def format_fraction_tag(fraction):
    return f"f{int(round(float(fraction) * 100)):02d}"


def build_run_name(base_name, adaptation_name, fraction):
    fraction_tag = format_fraction_tag(fraction)
    if base_name:
        return f"{base_name}_{adaptation_name}_{fraction_tag}"
    return f"{adaptation_name}_{fraction_tag}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run EA and AdaBN fraction sweeps for EEGNet.")
    parser.add_argument("--config", required=True, help="Base YAML/JSON experiment config.")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
        help="Fractions of target data used for adaptation.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Adaptation methods to sweep.",
    )
    parser.add_argument(
        "--summary-name",
        default="adaptation_fraction_sweep_summary.csv",
        help="Filename for the combined sweep summary CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = load_config(args.config)
    summary_rows = []

    for adaptation_name in args.methods:
        for fraction in args.fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"Invalid fraction {fraction}. Use values in the interval (0, 1].")

            run_config = replace(
                base_config,
                adaptation_name=adaptation_name,
                adaptation_target_fraction=float(fraction),
                run_name=build_run_name(base_config.run_name, adaptation_name, fraction),
            )
            fold_metrics, _, summary = run_experiment(run_config)
            summary_row = summary.iloc[0].to_dict() if not summary.empty else {}
            summary_row.update(
                {
                    "adaptation_name": adaptation_name,
                    "adaptation_target_fraction": float(fraction),
                    "run_name": run_config.run_name,
                    "n_folds": int(len(fold_metrics)),
                }
            )
            summary_rows.append(summary_row)

    if summary_rows:
        summary_frame = pd.DataFrame(summary_rows)
        summary_path = Path(base_config.output_dir) / args.summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(summary_path, index=False)
        print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()