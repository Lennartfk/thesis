import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Suppress MNE info messages
mne.set_log_level("WARNING")

def get_covariances(X):
    covs = []
    for i in range(X.shape[0]):
        trial = X[i]
        trial = trial - np.mean(trial, axis=1, keepdims=True)
        cov = np.dot(trial, trial.T) / (trial.shape[1] - 1)
        covs.append(cov)
    return np.array(covs)

def extract_features(covs):
    features = []
    for cov in covs:
        idx = np.triu_indices_from(cov)
        features.append(cov[idx])
    return np.array(features)

def generate_scatter_plot(out_path="data/results/Final_Plots_All/Feature_Visualizations/ea_scatter_real_data_2d.pdf", include_title=True):
    print("Loading data for 5 subjects...")
    epoch_dir = Path("data/SEED_VIG/epoched")
    
    # Pick 5 distinct subjects manually to ensure variety
    target_subs = ["1_20151124_noon_2-epo.fif", "2_20151106_noon-epo.fif", "6_20151121_noon-epo.fif", "8_20151022_noon-epo.fif", "12_20150928_noon-epo.fif"]
    selected_files = [epoch_dir / f for f in target_subs]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    all_feat_raw = []
    all_feat_ea = []
    
    labels = []
    
    for i, path in enumerate(selected_files):
        print(f"Processing {path.name}...")
        epochs = mne.read_epochs(path, preload=True, verbose=False).get_data(copy=False)
        np.random.seed(42)
        idx = np.random.choice(epochs.shape[0], 200, replace=False)
        X = epochs[idx]
        
        # Raw Covariances
        cov = get_covariances(X)
        feat = extract_features(cov)
        
        # EA Alignment
        R = np.mean(cov, axis=0)
        R_inv_sqrt = np.real(sqrtm(inv(R))) # Real to avoid complex casting warnings
        X_ea = np.array([np.dot(R_inv_sqrt, x) for x in X])
        cov_ea = get_covariances(X_ea)
        feat_ea = extract_features(cov_ea)
        
        all_feat_raw.append(feat)
        all_feat_ea.append(feat_ea)
        labels.extend([i] * 200)
        
    X_raw = np.vstack(all_feat_raw)
    X_ea = np.vstack(all_feat_ea)
    
    # Crucial: Standardize features before PCA to prevent one subject's variance from dominating
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw)
    
    scaler_ea = StandardScaler()
    X_ea_scaled = scaler_ea.fit_transform(X_ea)
    
    print("Running PCA...")
    pca_raw = PCA(n_components=2)
    X_raw_2d = pca_raw.fit_transform(X_raw_scaled)
    
    pca_ea = PCA(n_components=2)
    X_ea_2d = pca_ea.fit_transform(X_ea_scaled)
    
    # Plotting
    print("Plotting...")
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    for i in range(5):
        mask = np.array(labels) == i
        ax1.scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1], alpha=0.6, label=f'Subject {i+1}', c=colors[i], s=40)
        ax2.scatter(X_ea_2d[mask, 0], X_ea_2d[mask, 1], alpha=0.6, label=f'Subject {i+1}', c=colors[i], s=40)
        
    if include_title:
        ax1.set_title("Before EA", fontsize=16, fontweight='bold')
        ax2.set_title("After EA", fontsize=16, fontweight='bold')
    
    ax1.legend()
    ax2.legend()
    
    # Match axes perfectly without forcing a square range
    all_data = np.vstack([X_raw_2d, X_ea_2d])
    
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)
    
    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    generate_scatter_plot()
