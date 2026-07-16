import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from src.models.deep.eegnet import build_eegnet
from src.models.adaptation.adabn import adapt_batch_norm
import mne

def plot_adabn_tsne(ckpt_path, target_subject, source_subjects, epochs_dir, out_path):
    """
    Plots a before/after t-SNE scatter of the latent space to show 
    how AdaBN forces the Target subject domain to merge with the Source domains.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load model
    model = build_eegnet(n_channels=17, n_samples=1600)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    # Load data
    def load_subject_data(sub_id):
        pattern = f"{sub_id}_*-epo.fif"
        import glob
        matches = glob.glob(os.path.join(epochs_dir, pattern))
        if not matches:
            return None
        epochs = mne.read_epochs(matches[0], preload=True, verbose=False)
        return torch.tensor(epochs.get_data(copy=False), dtype=torch.float32)

    X_target = load_subject_data(target_subject)
    if X_target is None:
        raise ValueError(f"Could not load data for target {target_subject}")
        
    X_sources = []
    source_labels = []
    for s in source_subjects:
        xs = load_subject_data(s)
        if xs is not None:
            # We subsample to avoid crowding the plot
            # Just take 300 random epochs per source
            idx = np.random.choice(len(xs), min(300, len(xs)), replace=False)
            X_sources.append(xs[idx])
            source_labels.extend([s] * len(idx))
    
    # Subsample target
    idx_t = np.random.choice(len(X_target), min(400, len(X_target)), replace=False)
    X_target_sub = X_target[idx_t]
    
    X_all_sources = torch.cat(X_sources, dim=0)

    # Helper to get features
    def get_features(m, x):
        with torch.no_grad():
            m.eval()
            if x.ndim == 3:
                x = x.unsqueeze(1)
            return m._forward_features(x).numpy()

    feat_target_before = get_features(model, X_target_sub)
    feat_source_before = get_features(model, X_all_sources)
    
    class Config:
        device = 'cpu'
        batch_size = 64
        num_workers = 0
    print("Adapting Batch Norm to Target...")
    adapt_batch_norm(model, X_target.numpy(), Config())
    
    feat_target_after = get_features(model, X_target_sub)
    
    all_feat = np.vstack([feat_source_before, feat_target_before, feat_target_after])
    print("Computing unified t-SNE...")
    tsne_all = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_feat)
    
    n_src = len(feat_source_before)
    n_tgt = len(feat_target_before)
    
    tsne_source = tsne_all[:n_src]
    tsne_target_before = tsne_all[n_src:n_src+n_tgt]
    tsne_target_after = tsne_all[n_src+n_tgt:]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Latent Feature Space (t-SNE)', fontsize=18)
    
    n_tgt = len(feat_target_before)
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (tsne_tgt, title) in enumerate(zip([tsne_target_before, tsne_target_after], ["Before AdaBN", "After AdaBN"])):
        ax = axes[i]
        
        # Plot Sources
        curr_idx = 0
        for j, s in enumerate(source_subjects):
            count = source_labels.count(s)
            c = colors[j % len(colors)]
            ax.scatter(tsne_source[curr_idx:curr_idx+count, 0], tsne_source[curr_idx:curr_idx+count, 1], 
                       c=c, alpha=0.3, label=f'Source (Sub {s})', s=15, zorder=1)
            curr_idx += count
            
        # Plot Target
        ax.scatter(tsne_tgt[:, 0], tsne_tgt[:, 1], c='red', alpha=0.9, edgecolors='black', linewidth=0.5, label=f'Target (Sub {target_subject})', s=25, zorder=5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved AdaBN t-SNE plot to {out_path}")
