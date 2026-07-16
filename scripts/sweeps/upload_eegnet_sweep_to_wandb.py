"""Upload completed EEGNet sweep runs to Weights & Biases for interactive exploration.

Usage:
  1. Install: `pip install wandb pyyaml pandas`
  2. Login: `wandb login` (or set `WANDB_API_KEY` env var)
  3. Run: `python scripts/upload_eegnet_sweep_to_wandb.py --project eegnet_sweep --entity <your-entity>`

The script creates one W&B run per completed trial, setting the run `config` to
the trial YAML and logging the `balanced_accuracy_mean` (and other summary metrics
if present). After uploading, open the W&B project and add a Parallel Coordinates
panel to explore hyperparameters interactively.
"""
from pathlib import Path
import argparse
import yaml
import pandas as pd
import sys


def extract_trial_index(run_name):
    import re

    m = re.search(r"_t(\d{3})_", run_name)
    return m.group(1) if m else None


def load_config(config_dir, run_name):
    idx = extract_trial_index(run_name)
    if idx is None:
        return {}
    path = Path(config_dir) / f"trial_{idx}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_runs(sweep_dir):
    root = Path(sweep_dir)
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        summary = p / "summary_metrics.csv"
        if not summary.exists():
            continue
        yield p


def read_summary_metrics(run_dir):
    path = Path(run_dir) / "summary_metrics.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    for k, v in list(row.items()):
        try:
            if pd.isna(v):
                row[k] = None
            elif hasattr(v, 'item'):
                row[k] = v.item()
        except Exception:
            pass
    return row


def main():
    parser = argparse.ArgumentParser(description="Upload EEGNet sweep runs to W&B")
    parser.add_argument("--sweep-dir", default="data/results/sweeps/eegnet_base")
    parser.add_argument("--config-dir", default="configs/sweeps/eegnet_base")
    parser.add_argument("--project", default="eegnet_sweep")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs to upload")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded but don't contact W&B")
    args = parser.parse_args()

    try:
        import wandb
    except Exception as exc:
        print("Please install wandb: pip install wandb", file=sys.stderr)
        raise SystemExit(1) from exc

    runs = list(find_runs(args.sweep_dir))
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        print(f"No completed runs found under {args.sweep_dir}")
        return

    print(f"Found {len(runs)} runs. Dry run: {args.dry_run}")

    for p in runs:
        run_name = p.name
        config = load_config(args.config_dir, run_name)
        summary = read_summary_metrics(p)

        metrics = {}
        for key in [
            "balanced_accuracy_mean",
            "f1_mean",
            "roc_auc_mean",
            "accuracy_mean",
        ]:
            if key in summary:
                metrics[key] = summary[key]

        def _fmt(v):
            try:
                if v is None:
                    return "None"
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return str(v)

        for key in (
            "learning_rate",
            "batch_size",
            "weight_decay",
            "eegnet_dropout",
            "eegnet_f1",
            "eegnet_depth_multiplier",
            "eegnet_temporal_kernel_length",
            "eegnet_separable_kernel_length",
            "tune_decision_threshold",
        ):
            sval = _fmt(config.get(key))
            config_key = f"{key}_str"
            if config_key not in config:
                config[config_key] = sval

        print(f"Preparing {run_name}: metrics={metrics}, config keys={list(config.keys())}")

        if args.dry_run:
            continue

        try:
            wandb.init(project=args.project, entity=args.entity, name=run_name, config=config, reinit=True)
        except Exception as exc:
            print(f"Failed to init with entity={args.entity}: {exc}. Retrying without entity...")
            wandb.init(project=args.project, name=run_name, config=config, reinit=True)
        if metrics:
            wandb.log(metrics)
        try:
            wandb.run.summary.update(metrics)
        except Exception:
            pass

        str_keys = [k for k in config.keys() if k.endswith("_str")]
        if str_keys:
            try:
                for sk in str_keys:
                    val = config.get(sk)
                    wandb.run.summary[sk] = None if val is None else str(val)
                wandb.run.summary.update({sk: wandb.run.summary[sk] for sk in str_keys})
            except Exception:
                pass
        wandb.finish()

    print("Upload complete. Open W&B project and create a Parallel Coordinates panel to explore.")


if __name__ == "__main__":
    main()
