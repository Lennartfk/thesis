# EEG Domain Adaptation Research

Generalization across subjects using domain adaptation techniques with EEG data and the SEED-VIG dataset.

## Project Structure

- **data/** - Raw and processed EEG data
  - SEED_VIG/raw/ - Original SEED-VIG dataset
  - SEED_VIG/processed/ - Filtered MNE raw FIF files
  - SEED_VIG/epoched/ - Fixed-length epoch FIF files with PERCLOS metadata
  - SEED_VIG/features/ - One tabular feature CSV for modeling
- **notebooks/** - Jupyter notebooks for exploration and analysis
- **src/** - Python modules (preprocessing, models, utils)
- **models/** - Trained model checkpoints
- **results/** - Outputs, figures, evaluation metrics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download SEED-VIG dataset and place in `data/raw/`

## Usage

Build reusable preprocessing artifacts first:

```bash
python src/data/process_all_data.py
python src/data/epoch_seedvig.py
python src/features/extract_epoch_features.py
```

The feature extraction step writes `data/SEED_VIG/features/seedvig_spectral_features.csv` with:

```text
subject_id, recording, epoch_index, perclos, <per-channel band powers>, <optional mean/std>
```

PERCLOS labels are attached during epoching and are not reloaded during feature extraction or model training. Feature values are left unnormalized; each SVM uses an sklearn `Pipeline` with `StandardScaler` inside the LOSO training fold.

Run experiments through one MLflow-tracked entrypoint:

```bash
python -m src.experiments.run_experiment --model svm_rbf --adaptation none --seed 42
```

Useful registered names:

```text
Models: svm_linear, svm_rbf, logistic_regression, random_forest, eegnet
Adaptation: none, coral, tca
```

`eegnet` and `tca` are registry placeholders for future implementations; they use the same experiment schema once their fold-local training code is added.
