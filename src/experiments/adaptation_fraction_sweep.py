from dataclasses import replace
from pathlib import Path
import pandas as pd


def format_fraction_tag(fraction):
    return f"f{int(round(float(fraction) * 100)):02d}"


def build_run_name(base_name, adaptation_name, fraction):
    fraction_tag = format_fraction_tag(fraction)
    if base_name:
        return f"{base_name}_{adaptation_name}_{fraction_tag}"
    return f"{adaptation_name}_{fraction_tag}"


def run_fraction_sweep(base_config, fractions, methods, summary_name=None):
    """Run adaptation fraction sweep using the provided `run_experiment` entrypoint.

    Parameters
    - base_config: ExperimentConfig dataclass instance
    - fractions: iterable of floats in (0, 1]
    - methods: iterable of adaptation method names (e.g., ['ea', 'adabn'])
    - summary_name: optional filename to write combined summary to base_config.output_dir

    Returns a pandas.DataFrame with one row per run (summary rows collected).
    """
    from src.experiments.run_experiment import run_experiment

    summary_rows = []

    for adaptation_name in methods:
        for fraction in fractions:
            if not 0.0 < float(fraction) <= 1.0:
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

    summary_frame = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    if summary_frame is not None and summary_name:
        summary_path = Path(base_config.output_dir) / summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(summary_path, index=False)
        print(f"Saved sweep summary to {summary_path}")

    return summary_frame
