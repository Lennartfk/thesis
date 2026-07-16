# Unsupervised Domain Adaptation for EEG-based Fatigue Detection

This repository contains the complete codebase for the Bachelor's thesis evaluating unsupervised domain adaptation methods on EEG data. Specifically, we investigate whether techniques like **Euclidean Alignment (EA)** and **Adaptive Batch Normalization (AdaBN)** can improve the cross-subject generalization of a deep learning model (EEGNet) for mental fatigue detection using the publicly available SEED-VIG dataset.

A key contribution of this repository is the evaluation of these methods under realistic, strictly chronological calibration protocols:
- **Random Sampling (Idealized)**: Random subsets of target data (often violating causality).
- **Chronological Calibration**: Using the first $N$ minutes of a subject's recording.
- **Sequential Window Calibration**: Sliding a fixed-size calibration window continuously over the data.

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv thesis_env
   source thesis_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the Dataset**
   - Download the raw **SEED-VIG** dataset.
   - Place the raw `.mat` files in `data/SEED_VIG/raw/`.

## 🔄 Data Preprocessing

To replicate the data preprocessing pipeline (which filters the raw data, epochs it into 8-second segments, and extracts features):

```bash
python src/data/process_all_data.py
python src/data/epoch_seedvig.py
python src/features/extract_epoch_features.py
```
*Note: The intermediate PERCLOS epochs ($0.35 < \text{PERCLOS} < 0.70$) are dropped during evaluation but retained in the metadata.*

## 🚀 Training the Baseline EEGNet

To train the standard EEGNet model using Leave-One-Subject-Out (LOSO) cross-validation (without any domain adaptation):

```bash
# Option 1: Via Python
python -m src.experiments.run_experiment --config configs/eegnet_baseline.yaml

# Option 2: Via SLURM (if on a cluster)
sbatch scripts/slurm/final_train_baseline.slurm
```
This will train the baseline model and log results to the `mlruns/` directory using MLflow.

## 🎛️ Running Domain Adaptation Calibration Protocols

Once the baseline is established, you can evaluate the domain adaptation methods (EA and AdaBN) using the three calibration protocols. We sweep across different calibration durations ($N \in \{5, 10, 15, 20, 30, 45, 60, 90\}$ minutes).

**1. Random Sampling (Idealized)**
```bash
sbatch scripts/slurm/final_eval_fraction.slurm
```

**2. Chronological Calibration**
```bash
sbatch scripts/slurm/final_eval_chronological.slurm
```

**3. Sequential Window Calibration**
```bash
sbatch scripts/slurm/evaluate_sequential_window.slurm
```
*Note: SLURM scripts are configured to use Slurm Array Jobs for parallel execution. You can inspect the SLURM files for the exact python commands to run them locally.*

## 📊 Reproducing Figures and Statistical Analysis

All figures and statistical analyses used in the thesis can be generated automatically from the experimental outputs. 

**Generate Statistics (Paired T-Tests with FDR correction)**
```bash
python scripts/stats_adaptation_fraction.py
```

**Generate Vector Graphics (PDFs)**
To generate all plots (including chronological decays, 3D accuracy surfaces, adaptation gains, and topological maps):
```bash
sbatch scripts/slurm/generate_all_pdfs.slurm
```
*This script runs `scripts/generate_thesis_figures.py`, which orchestrates all plotting scripts and saves the final PDFs into `data/results/Thesis_Figures/`.*

## 📁 Repository Structure
- `configs/` - YAML configuration files for experiments.
- `data/` - Dataset and evaluation results (ignored in version control).
- `scripts/` - Slurm submission scripts, plotting routines, and statistical analysis scripts.
- `src/` - Core Python modules:
  - `data/` - Loading, epoching, and filtering.
  - `models/` - EEGNet architecture and AdaBN implementation.
  - `experiments/` - Cross-validation logic and calibration protocol loops.
  - `features/` - Euclidean Alignment and spatial covariance matrices.
  - `training/` - PyTorch training loops.
  - `visualization/` - Core plotting functions.
