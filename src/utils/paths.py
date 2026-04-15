from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SEED_VIG_DIR = DATA_DIR / "SEED_VIG"
RESULTS_DIR = DATA_DIR / "results"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DEFAULT_FEATURE_PATH = SEED_VIG_DIR / "features" / "seedvig_spectral_features.csv"
DEFAULT_EXPERIMENT_OUTPUT_DIR = RESULTS_DIR / "experiments"
