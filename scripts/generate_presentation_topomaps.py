#!/usr/bin/env python3
"""Generate Alert vs. Drowsy MNE Topomaps for presentation."""
import os
import glob
import mne
import argparse

from src.utils.mne_topomaps import plot_alert_vs_drowsy_topomaps


def get_subject_id(filename):
    # e.g., "10_20151125_noon-epo.fif" -> "10"
    base = os.path.basename(filename)
    return base.split("_")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-dir", default="data/SEED_VIG/epoched", help="Path to epoch files")
    p.add_argument("--out-dir", default="data/results/Final_Plots_All/Topomaps", help="Output dir for PDFs")
    p.add_argument("--no-title", action="store_true", help="Disable titles")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    epoch_files = sorted(glob.glob(os.path.join(args.epochs_dir, "*-epo.fif")))
    if not epoch_files:
        print(f"No epoch files found in {args.epochs_dir}")
        return

    all_alert_epochs = []
    all_drowsy_epochs = []

    print("Loading epochs and generating individual subject topomaps...")
    for fpath in epoch_files:
        subject_id = get_subject_id(fpath)
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        
        # In SEED-VIG, PERCLOS is typically stored in epochs.metadata['PERCLOS']
        # Let's filter: Alert < 0.35, Drowsy > 0.70
        if epochs.metadata is None or 'perclos' not in epochs.metadata.columns:
            print(f"Skipping {subject_id}: no perclos metadata.")
            continue
            
        alert_mask = epochs.metadata['perclos'] < 0.35
        drowsy_mask = epochs.metadata['perclos'] > 0.70
        
        epochs_alert = epochs[alert_mask]
        epochs_drowsy = epochs[drowsy_mask]
        
        if len(epochs_alert) == 0 or len(epochs_drowsy) == 0:
            print(f"Skipping subject {subject_id}: insufficient alert/drowsy epochs.")
            continue
            
        # Accumulate for global average
        all_alert_epochs.append(epochs_alert)
        all_drowsy_epochs.append(epochs_drowsy)

        # Plot individual subject
        print(f"Plotting Subject {subject_id}...")
        plot_alert_vs_drowsy_topomaps(
            epochs_alert, 
            epochs_drowsy, 
            title=f"Alert vs. Drowsy Power Spectral Density (Subject {subject_id})",
            out_path=os.path.join(args.out_dir, f"topomap_subject_{subject_id}.pdf"),
            include_title=not args.no_title
        )

    if all_alert_epochs and all_drowsy_epochs:
        print("Generating Global SEED-VIG topomap...")
        # Concatenate all subjects
        global_alert = mne.concatenate_epochs(all_alert_epochs, verbose=False)
        global_drowsy = mne.concatenate_epochs(all_drowsy_epochs, verbose=False)
        
        plot_alert_vs_drowsy_topomaps(
            global_alert, 
            global_drowsy, 
            title="Alert vs. Drowsy Power Spectral Density (Global SEED-VIG Average)",
            out_path=os.path.join(args.out_dir, "topomap_global_average.pdf"),
            include_title=not args.no_title
        )
        print(f"Finished! Plots saved to {args.out_dir}")
    else:
        print("Could not generate global topomap due to lack of valid epochs.")

if __name__ == "__main__":
    main()
