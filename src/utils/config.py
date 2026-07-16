import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.utils.paths import DEFAULT_EPOCH_DIR, DEFAULT_EXPERIMENT_OUTPUT_DIR, DEFAULT_FEATURE_PATH, MLRUNS_DIR


@dataclass
class ExperimentConfig:
    dataset_name: str = "SEED-VIG"
    feature_path: str = str(DEFAULT_FEATURE_PATH)
    epoch_dir: str = str(DEFAULT_EPOCH_DIR)
    output_dir: str = str(DEFAULT_EXPERIMENT_OUTPUT_DIR)
    mlflow_tracking_uri: str = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
    mlflow_experiment_name: str = "seedvig_loso"
    run_name: str = ""
    feature_set_name: str = "spectral_bandpower_mean_std"
    preprocessing_version: str = "seedvig_raw_filter_1_50_epoch_8s_metadata_v1"
    model_family: str = "sklearn"
    input_type: str = "features"
    model_name: str = "svm_rbf"
    adaptation_name: str = "none"
    seed: int = 42
    alert_threshold: float = 0.35
    drowsy_threshold: float = 0.70
    eval_protocol: str = "leave_one_subject_out"
    loso_protocol: str = "leave_one_subject_out"
    positive_label: int = 1
    exclude_imbalanced_subjects: bool = False
    min_subject_class_samples: int = 1
    save_predictions: bool = True
    save_plots: bool = True
    save_checkpoints: bool = True
    log_models: bool = True
    register_model: bool = False
    registry_model_name: str = ""
    validation_subject_strategy: str = "cyclic_next_subject"
    within_subject_split_strategy: str = "stratified"
    val_min_class_samples: int = 10
    normalization: str = "channel_standard"
    early_stopping_metric: str = "loss"
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    patience: int = 15
    device: str = "auto"
    num_workers: int = 0
    use_class_weights: bool = True
    tune_decision_threshold: bool = False
    threshold_metric: str = "balanced_accuracy"
    decision_threshold: float = 0.5
    adaptation_target_fraction: float = 1.0
    adaptation_fraction_sweep: bool = False
    adaptation_fraction_cv_sweep: bool = False
    adaptation_fraction_sweep_fractions: str = ""
    adaptation_fraction_sweep_methods: str = ""
    adaptation_fraction_sweep_summary_name: str = "adaptation_fraction_sweep_summary.csv"
    chronological_sweep: bool = False
    fixed_window_sweep: bool = False
    sequential_window_sweep: bool = False
    baseline_run_name: str = "Baselines/Deep/Cross-subject/EEGNET_Baseline"
    ea_baseline_run_name: str = "Adaptation/EA/ea_f1.0"
    eegnet_f1: int = 8
    eegnet_depth_multiplier: int = 2
    eegnet_dropout: float = 0.5
    eegnet_temporal_kernel_length: int = 64
    eegnet_separable_kernel_length: int = 16
    eegnet_pool1_kernel: int = 4
    eegnet_pool2_kernel: int = 8


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
    parser.add_argument("--adaptation", dest="adaptation_name", help="Registered adaptation name.")
    parser.add_argument("--features", dest="feature_path", help="Feature CSV path.")
    parser.add_argument("--epoch-dir", dest="epoch_dir", help="Directory containing epoched FIF files.")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for result artifacts.")
    parser.add_argument("--run-name", dest="run_name", help="MLflow run name.")
    parser.add_argument("--baseline-run-name", dest="baseline_run_name", help="Run name of the standard baseline model.")
    parser.add_argument("--ea-baseline-run-name", dest="ea_baseline_run_name", help="Run name of the EA-aligned baseline model.")
    parser.add_argument("--mlflow-tracking-uri", dest="mlflow_tracking_uri", help="MLflow tracking URI (e.g., sqlite:////path/to/mlflow.db).")
    parser.add_argument("--mlflow-experiment-name", dest="mlflow_experiment_name", help="MLflow experiment name.")
    parser.add_argument("--model-family", dest="model_family", choices=["sklearn", "deep"], help="Experiment backend.")
    parser.add_argument("--input-type", dest="input_type", choices=["features", "epochs"], help="Input representation.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--max-epochs", type=int, help="Maximum deep-learning epochs.")
    parser.add_argument("--patience", type=int, help="Early-stopping patience for deep models.")
    parser.add_argument("--early-stopping-metric", dest="early_stopping_metric", help="Metric to use for early stopping (e.g., loss, balanced_accuracy, roc_auc, f1).")
    parser.add_argument("--batch-size", type=int, help="Batch size for deep models.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for deep models.")
    parser.add_argument(
        "--tune-decision-threshold",
        dest="tune_decision_threshold",
        action="store_true",
        help="Tune the binary decision threshold on the validation split for deep models.",
    )
    parser.add_argument(
        "--fixed-decision-threshold",
        dest="tune_decision_threshold",
        action="store_false",
        help="Use --decision-threshold directly instead of validation threshold tuning.",
    )
    parser.set_defaults(tune_decision_threshold=None)
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
        help="Validation metric maximized when tuning the binary decision threshold.",
    )
    parser.add_argument("--decision-threshold", type=float, help="Fixed/fallback drowsy-score threshold.")
    parser.add_argument(
        "--adaptation-target-fraction",
        type=float,
        help="Fraction of held-out target epochs used for unlabeled adaptation.",
    )
    parser.add_argument(
        "--fraction-sweep",
        dest="adaptation_fraction_sweep",
        action="store_true",
        help="Run adaptation fraction sweep over multiple fractions and methods.",
    )
    parser.add_argument("--chronological-sweep", action="store_true")
    parser.add_argument("--fixed-window-sweep", action="store_true")
    parser.add_argument("--sequential-window-sweep", action="store_true")
    parser.add_argument("--adaptation-fraction-cv-sweep", action="store_true", help="Run 10-fold CV sweep over adaptation fractions without training")
    parser.add_argument(
        "--sweep-fractions",
        dest="adaptation_fraction_sweep_fractions",
        help="Comma-separated fractions to use for the sweep, e.g. '0.1,0.25,0.5'.",
    )
    parser.add_argument(
        "--sweep-methods",
        dest="adaptation_fraction_sweep_methods",
        help="Comma-separated adaptation methods to sweep, e.g. 'ea,adabn'.",
    )
    parser.add_argument(
        "--sweep-summary-name",
        dest="adaptation_fraction_sweep_summary_name",
        help="Filename for the combined sweep summary CSV (written to output_dir).",
    )
    parser.add_argument(
        "--eval-protocol",
        dest="eval_protocol",
        choices=["leave_one_subject_out", "within_subject"],
        help="Evaluation protocol: LOSO or within-subject.",
    )
    parser.add_argument(
        "--within-subject-split-strategy",
        dest="within_subject_split_strategy",
        choices=["stratified", "chronological"],
        help="Within-subject split strategy. Stratified is recommended for binary classification.",
    )
    parser.add_argument(
        "--validation-subject-strategy",
        dest="validation_subject_strategy",
        choices=["cyclic_next_subject", "previous_subject", "random_subject"],
        help="Validation subject strategy for deep LOSO runs.",
    )
    parser.add_argument(
        "--val-min-class-samples",
        dest="val_min_class_samples",
        type=int,
        help="Minimum number of alert and drowsy samples required for a validation subject.",
    )
    parser.add_argument("--alert-threshold", type=float, help="PERCLOS alert threshold.")
    parser.add_argument("--drowsy-threshold", type=float, help="PERCLOS drowsy threshold.")
    parser.add_argument(
        "--exclude-imbalanced-subjects",
        dest="exclude_imbalanced_subjects",
        action="store_true",
        help="Exclude subjects whose retained alert or drowsy count is below --min-subject-class-samples.",
    )
    parser.add_argument(
        "--include-imbalanced-subjects",
        dest="exclude_imbalanced_subjects",
        action="store_false",
        help="Keep subjects even if one retained class has too few samples.",
    )
    parser.set_defaults(exclude_imbalanced_subjects=None)
    parser.add_argument(
        "--min-subject-class-samples",
        type=int,
        help="Minimum retained samples required in each class per subject when filtering imbalanced subjects.",
    )
    parser.add_argument(
        "--register-model",
        dest="register_model",
        action="store_true",
        help="Register the best fold model into MLflow Model Registry.",
    )
    parser.add_argument(
        "--registry-model-name",
        dest="registry_model_name",
        help="Optional MLflow Model Registry name. Defaults to dataset_model_adaptation.",
    )
    args = parser.parse_args()

    overrides = {
        "model_name": args.model_name,
        "adaptation_name": args.adaptation_name,
        "feature_path": args.feature_path,
        "epoch_dir": args.epoch_dir,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "baseline_run_name": args.baseline_run_name,
        "ea_baseline_run_name": args.ea_baseline_run_name,
        "model_family": args.model_family,
        "input_type": args.input_type,
        "seed": args.seed,
        "eval_protocol": args.eval_protocol,
        "within_subject_split_strategy": args.within_subject_split_strategy,
        "validation_subject_strategy": args.validation_subject_strategy,
        "val_min_class_samples": args.val_min_class_samples,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stopping_metric": args.early_stopping_metric,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "tune_decision_threshold": args.tune_decision_threshold,
        "threshold_metric": args.threshold_metric,
        "decision_threshold": args.decision_threshold,
        "adaptation_target_fraction": args.adaptation_target_fraction,
        "alert_threshold": args.alert_threshold,
        "drowsy_threshold": args.drowsy_threshold,
        "exclude_imbalanced_subjects": args.exclude_imbalanced_subjects,
        "min_subject_class_samples": args.min_subject_class_samples,
        "register_model": args.register_model,
        "registry_model_name": args.registry_model_name,
        "adaptation_fraction_sweep": args.adaptation_fraction_sweep,
        "adaptation_fraction_cv_sweep": getattr(args, "adaptation_fraction_cv_sweep", False),
        "adaptation_fraction_sweep_fractions": args.adaptation_fraction_sweep_fractions,
        "adaptation_fraction_sweep_methods": args.adaptation_fraction_sweep_methods,
        "adaptation_fraction_sweep_summary_name": args.adaptation_fraction_sweep_summary_name,
        "chronological_sweep": getattr(args, "chronological_sweep", False),
        "fixed_window_sweep": getattr(args, "fixed_window_sweep", False),
        "sequential_window_sweep": getattr(args, "sequential_window_sweep", False),
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment_name": args.mlflow_experiment_name,
    }
    return load_config(args.config, overrides=overrides)
