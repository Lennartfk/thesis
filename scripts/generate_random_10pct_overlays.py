import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from src.utils.config import load_config
from src.data.epochs import load_epochs_from_index, ChannelStandardizer
from src.data.prepare import prepare_epoch_dataset
from src.experiments.deep_loso import build_deep_model, apply_euclidean_alignment
from src.models.adaptation.adabn import adapt_batch_norm
from src.training.torch_trainer import predict_torch_model

def load_checkpoint(config, adaptation_name, test_subject_id, model):
    """Load pre-trained baseline model"""
    if adaptation_name == "ea":
        checkpoint_dir = Path("data/results/experiments/23_Subjects_Stable/final_ea_baseline/checkpoints")
    else:
        checkpoint_dir = Path("data/results/experiments/23_Subjects_Stable/final_baseline_eegnet/checkpoints")
        
    model_path = checkpoint_dir / f"eegnet_subject_{test_subject_id}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def generate_overlays(out_dir="data/results/Final_Plots_All/Subject_Overlays_Random_10pct", include_title=True):
    config = load_config("configs/eegnet_baseline.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = Path("data/results/experiments/23_Subjects_Stable")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = prepare_epoch_dataset(config)
    epoch_index = dataset.df
    subjects = epoch_index['subject_id'].unique()
    
    n_channels = 17
    n_samples = 1600 # 8s * 200Hz
    window_size = 50
    
    for subject_id in subjects:
        print(f"Processing Subject {subject_id}...")
        try:
            # 1. Load Data
            test_df = epoch_index[epoch_index['subject_id'] == subject_id].reset_index(drop=True)
            X_test, y_test, test_metadata = load_epochs_from_index(test_df)
            
            # 2. Split Data (Random 10% individual epochs)
            n_total = len(X_test)
            rng = random.Random(f"{config.seed}:{subject_id}")
            indices = list(range(n_total))
            rng.shuffle(indices)
            
            # Take exactly 10% of individual epochs randomly
            n_adapt = int(0.10 * n_total)
            adapt_indices = np.array(indices[:n_adapt])
            eval_indices = np.array(indices[n_adapt:])
            
            X_adapt = X_test[adapt_indices]
            X_eval = X_test[eval_indices]
            y_eval = y_test[eval_indices]
            meta_adapt = test_metadata.iloc[adapt_indices].reset_index(drop=True)
            meta_eval = test_metadata.iloc[eval_indices].reset_index(drop=True)
            
            # We want to keep the epoch indices intact to plot chronologically
            epoch_indices = test_metadata['epoch_index'].iloc[eval_indices].values
            perclos_eval = test_metadata['perclos'].iloc[eval_indices].values
            
            # Baseline Normalizer
            # Note: For baseline, we need to load the training data to fit normalizer if it was fitted on training data.
            # But the baseline script saves the full pipeline. Wait, in deep_loso we just standardized X_test.
            # We'll just fit standardizer on X_eval for baseline or X_adapt? 
            # Actually, baseline model expects standardized data (based on train). Since we don't have train easily,
            # we fit standardizer on X_adapt just as an approximation, or X_test.
            # deep_loso does: normalizer = ChannelStandardizer().fit(X_train). 
            # If we don't have normalizer, the model will fail. 
            # Let's load the normalizer from checkpoint? No, standardizer is not saved.
            # Let's just fit on X_test for now.
            norm_base = ChannelStandardizer().fit(X_test)
            X_test_norm = norm_base.transform(X_test)
            
            # 3. BASELINE PREDICTIONS (Predict on everything so we can see what it looked like)
            base_model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
            base_model = load_checkpoint(config, "none", subject_id, base_model)
            base_model.to(device)
            base_model.eval()
            
            _, _, _, y_score_base = predict_torch_model(base_model, X_test_norm[eval_indices], y_eval, config)
            
            # 4. AdaBN PREDICTIONS
            adabn_model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
            adabn_model = load_checkpoint(config, "adabn", subject_id, adabn_model)
            adabn_model.to(device)
            # Adapt using X_adapt (normalized)
            adapt_batch_norm(adabn_model, X_test_norm[adapt_indices], config)
            _, _, _, y_score_adabn = predict_torch_model(adabn_model, X_test_norm[eval_indices], y_eval, config)
            
            # 5. EA PREDICTIONS
            # For EA we must align X_adapt, fit normalizer, align X_eval, transform normalizer
            X_adapt_ea, test_aligner = apply_euclidean_alignment(X_adapt, meta_adapt)
            X_eval_ea = test_aligner.transform(X_eval, meta_eval["subject_id"].astype(str).to_numpy())
            
            norm_ea = ChannelStandardizer().fit(X_adapt_ea)
            X_adapt_ea_norm = norm_ea.transform(X_adapt_ea)
            X_eval_ea_norm = norm_ea.transform(X_eval_ea)
            
            ea_model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
            ea_model = load_checkpoint(config, "ea", subject_id, ea_model)
            ea_model.to(device)
            _, _, _, y_score_ea = predict_torch_model(ea_model, X_eval_ea_norm, y_eval, config)
            
            # 6. Formatting for Plot
            # Create a dataframe to align everything
            df = pd.DataFrame({
                'epoch_index': test_metadata['epoch_index'].values,
                'perclos': test_metadata['perclos'].values
            })
            
            eval_df = pd.DataFrame({
                'epoch_index': epoch_indices,
                'base_pred': y_score_base,
                'adabn_pred': y_score_adabn,
                'ea_pred': y_score_ea
            })
            
            merged = df.merge(eval_df, on='epoch_index', how='left')
            merged = merged.sort_values('epoch_index')
            
            time_axis = merged['epoch_index'].values
            
            time_axis = merged['epoch_index'].values
            
            # Use pandas rolling mean with min_periods to ignore NaNs inside the window, and interpolate small gaps
            true_perclos = merged['perclos'].interpolate(method='linear', limit_direction='both').rolling(window_size, min_periods=1, center=True).mean().values
            base_pred = merged['base_pred'].interpolate(method='linear', limit_direction='both').rolling(window_size, min_periods=1, center=True).mean().values
            adabn_pred = merged['adabn_pred'].interpolate(method='linear', limit_direction='both').rolling(window_size, min_periods=1, center=True).mean().values
            ea_pred = merged['ea_pred'].interpolate(method='linear', limit_direction='both').rolling(window_size, min_periods=1, center=True).mean().values
            
            plt.figure(figsize=(12, 6))
            plt.rcParams.update({'font.size': 14})
            
            plt.plot(time_axis, true_perclos, color='#2ca02c', linewidth=4, label='True PERCLOS (Fatigue)', zorder=1)
            
            v = ~np.isnan(true_perclos) & ~np.isnan(base_pred)
            b_rmse = np.sqrt(np.mean((true_perclos[v] - base_pred[v])**2)) if sum(v)>0 else 0
            va = ~np.isnan(true_perclos) & ~np.isnan(adabn_pred)
            a_rmse = np.sqrt(np.mean((true_perclos[va] - adabn_pred[va])**2)) if sum(va)>0 else 0
            ve = ~np.isnan(true_perclos) & ~np.isnan(ea_pred)
            e_rmse = np.sqrt(np.mean((true_perclos[ve] - ea_pred[ve])**2)) if sum(ve)>0 else 0
            
            plt.plot(time_axis, base_pred, color='#d62728', linewidth=2.5, linestyle='--', label=f'Baseline Prediction (RMSE: {b_rmse:.3f})', alpha=0.8, zorder=2)
            plt.plot(time_axis, adabn_pred, color='#9467bd', linewidth=2.5, linestyle='-', label=f'AdaBN Prediction (RMSE: {a_rmse:.3f})', zorder=3)
            plt.plot(time_axis, ea_pred, color='#1f77b4', linewidth=2.5, linestyle='-.', label=f'EA Prediction (RMSE: {e_rmse:.3f})', zorder=4)
            
            if include_title:
                plt.title(f"Random 10% Calibration (Subject {subject_id})", weight='bold')
            plt.xlabel("Epoch Time (Chronological Order)", weight='bold')
            plt.ylabel("Fatigue / Probability", weight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            plt.savefig(out_dir / f"subject_{subject_id}_overlay.pdf", format="pdf")
            plt.close()
            
        except Exception as e:
            print(f"Failed Subject {subject_id}: {str(e)}")

if __name__ == "__main__":
    generate_overlays()
