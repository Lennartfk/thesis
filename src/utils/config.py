import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.utils.paths import DEFAULT_EXPERIMENT_OUTPUT_DIR, DEFAULT_FEATURE_PATH, MLRUNS_DIR


@dataclass
class ExperimentConfig:
    dataset_name: str = "SEED-VIG"
    feature_path: str = str(DEFAULT_FEATURE_PATH)
    output_dir: str = str(DEFAULT_EXPERIMENT_OUTPUT_DIR)
    mlflow_tracking_uri: str = str(MLRUNS_DIR)
    mlflow_experiment_name: str = "seedvig_loso"
    run_name: str = ""
    feature_set_name: str = "spectral_bandpower_mean_std"
    preprocessing_version: str = "seedvig_raw_filter_1_50_epoch_8s_metadata_v1"
    model_name: str = "svm_rbf"
    adaptation_name: str = "none"
    seed: int = 42
    alert_threshold: float = 0.35
    drowsy_threshold: float = 0.70
    loso_protocol: str = "leave_one_subject_out"
    positive_label: int = 1
    save_predictions: bool = True
    save_plots: bool = True


def _read_config_file(path):
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to read YAML config files, or use JSON.") from exc

    loaded = yaml.safe_load(text)
    return loaded or {}


def load_config(path=None, overrides=None):
    values = asdict(ExperimentConfig())
    if path:
        values.update(_read_config_file(path))
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})

    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown experiment config keys: {unknown}")

    return ExperimentConfig(**values)


def parse_experiment_args():
    parser = argparse.ArgumentParser(description="Run a SEED-VIG LOSO experiment with MLflow tracking.")
    parser.add_argument("--config", help="Optional JSON/YAML config file. Hydra is not used.")
    parser.add_argument("--model", dest="model_name", help="Registered model name, e.g. svm_rbf.")
    parser.add_argument("--adaptation", dest="adaptation_name", help="Registered adaptation name, e.g. none or coral.")
    parser.add_argument("--features", dest="feature_path", help="Feature CSV path.")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for result artifacts.")
    parser.add_argument("--run-name", dest="run_name", help="MLflow run name.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--alert-threshold", type=float, help="PERCLOS alert threshold.")
    parser.add_argument("--drowsy-threshold", type=float, help="PERCLOS drowsy threshold.")
    args = parser.parse_args()

    overrides = {
        "model_name": args.model_name,
        "adaptation_name": args.adaptation_name,
        "feature_path": args.feature_path,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "alert_threshold": args.alert_threshold,
        "drowsy_threshold": args.drowsy_threshold,
    }
    return load_config(args.config, overrides=overrides)
