import argparse
import itertools
import random
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SEARCH_SPACE = {
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [16, 30, 64],
    "weight_decay": [0.0, 0.0001, 0.001],
    "eegnet_dropout": [0.25, 0.5, 0.65],
    "eegnet_f1": [4, 8, 16],
    "eegnet_depth_multiplier": [1, 2, 4],
    "eegnet_temporal_kernel_length": [32, 64, 128],
    "eegnet_separable_kernel_length": [8, 16, 32],
    "tune_decision_threshold": [False, True],
}


FIXED_SWEEP_VALUES = {
    "adaptation_name": "none",
    "model_family": "deep",
    "input_type": "epochs",
    "model_name": "eegnet",
    "eval_protocol": "leave_one_subject_out",
    "loso_protocol": "leave_one_subject_out_random_subject_val",
    "validation_subject_strategy": "random_subject",
    "early_stopping_metric": "balanced_accuracy",
    "threshold_metric": "balanced_accuracy",
    "max_epochs": 60,
    "patience": 10,
    "register_model": False,
    "save_plots": False,
    "save_checkpoints": False,
    "log_models": False,
}


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def format_trial_name(index, values):
    threshold_tag = "thrval" if values["tune_decision_threshold"] else "thr050"
    return (
        f"sweep_eegnet_base_t{index:03d}"
        f"_lr{values['learning_rate']:g}"
        f"_bs{values['batch_size']}"
        f"_wd{values['weight_decay']:g}"
        f"_do{values['eegnet_dropout']:g}"
        f"_f1{values['eegnet_f1']}"
        f"_dm{values['eegnet_depth_multiplier']}"
        f"_tk{values['eegnet_temporal_kernel_length']}"
        f"_sk{values['eegnet_separable_kernel_length']}"
        f"_{threshold_tag}"
    )


def sample_trials(n_trials, seed):
    keys = list(SEARCH_SPACE)
    all_trials = [dict(zip(keys, values)) for values in itertools.product(*(SEARCH_SPACE[key] for key in keys))]
    rng = random.Random(seed)
    rng.shuffle(all_trials)
    return all_trials[: min(n_trials, len(all_trials))]


def write_configs(template, trials, output_dir, project_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []

    for index, values in enumerate(trials):
        config = dict(template)
        config.update(FIXED_SWEEP_VALUES)
        config.update(values)
        config["run_name"] = format_trial_name(index, values)
        config["output_dir"] = "data/results/sweeps/eegnet_base"
        config["mlflow_tracking_uri"] = f"sqlite:///{project_root / 'mlruns_sweeps' / 'eegnet_base' / f'trial_{index:03d}.db'}"
        config["mlflow_experiment_name"] = "seedvig_eegnet_base_sweep"

        path = output_dir / f"trial_{index:03d}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            try:
                yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
            except TypeError:
                yaml.safe_dump(config, handle, default_flow_style=False)
        config_paths.append(path)

    list_path = output_dir / "configs.txt"
    with list_path.open("w", encoding="utf-8") as handle:
        for path in config_paths:
            handle.write(f"{path.relative_to(project_root)}\n")

    return config_paths, list_path


def main():
    parser = argparse.ArgumentParser(description="Generate base EEGNet hyperparameter sweep YAML files.")
    parser.add_argument("--template", default="configs/eegnet_none_cross_lr001_bs30_do025_random.yaml")
    parser.add_argument("--output-dir", default="configs/sweeps/eegnet_base")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    template_path = PROJECT_ROOT / args.template
    output_dir = PROJECT_ROOT / args.output_dir
    template = load_yaml(template_path)
    trials = sample_trials(args.n_trials, args.seed)
    config_paths, list_path = write_configs(template, trials, output_dir, PROJECT_ROOT)

    print(f"Wrote {len(config_paths)} configs to {output_dir}")
    print(f"Wrote array list to {list_path}")


if __name__ == "__main__":
    main()
