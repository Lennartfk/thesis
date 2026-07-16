import torch
import numpy as np
import mne
import matplotlib.pyplot as plt
import os

from src.models.deep.eegnet import build_eegnet

def plot_eegnet_spatial_filters(checkpoint_dir, out_path, n_channels=17, n_samples=1600):
    """
    Loads trained EEGNet checkpoints, extracts the Spatial Filter weights, 
    averages the absolute weights across all filters and folds, and plots 
    a topomap showing which physical channels the model relies on most.
    """
    ch_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 
                'P1', 'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
                
    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
    info.set_montage('standard_1020')

    all_spatial_weights = []

    # Iterate over all checkpoints in the directory
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith('eegnet_subject_') and file.endswith('.pt'):
                ckpt_path = os.path.join(root, file)
                model = build_eegnet(n_channels=n_channels, n_samples=n_samples)
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                spatial_conv = model.spatial_block[0]
                weights = spatial_conv.weight.data.clone().cpu()

                mean_abs_weights = weights.abs().mean(dim=0).squeeze()
                all_spatial_weights.append(mean_abs_weights.numpy())

    if not all_spatial_weights:
        raise ValueError(f"No EEGNet checkpoints found in {checkpoint_dir}")

    global_weights = np.mean(all_spatial_weights, axis=0)
    
    global_weights = (global_weights - global_weights.min()) / (global_weights.max() - global_weights.min())

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle('EEGNet Global Spatial Filter Importance', fontsize=16, y=1.02)
    
    im, _ = mne.viz.plot_topomap(
        global_weights, info, axes=ax, show=False,
        cmap='Reds', sphere='eeglab'
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Importance Weight', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved EEGNet Spatial Filters plot to {out_path}")
