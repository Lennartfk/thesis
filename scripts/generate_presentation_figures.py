#!/usr/bin/env python3
"""Unified executable script to generate all presentation figures."""

import argparse
import os
import glob
import mne

from src.visualization.model_weights import plot_eegnet_spatial_filters
from src.visualization.latent_space import plot_adabn_tsne
from src.utils.mne_topomaps import plot_alert_vs_drowsy_topomaps
from src.visualization.power_progression import plot_power_progression_over_time

def generate_ea_topomaps(epochs_dir, out_dir):
    print("Generating Euclidean Alignment topomaps...")
    epoch_files = sorted(glob.glob(os.path.join(epochs_dir, "*-epo.fif")))
    if not epoch_files:
        print("No epochs found.")
        return
        
    all_alert_epochs = []
    all_drowsy_epochs = []
    
    from src.models.adaptation.euclidean_alignment import EuclideanAlignment
    import numpy as np
    
    for fpath in epoch_files:
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        if epochs.metadata is None or 'perclos' not in epochs.metadata.columns:
            continue
            
        alert_mask = epochs.metadata['perclos'] < 0.35
        drowsy_mask = epochs.metadata['perclos'] > 0.70
        
        if alert_mask.sum() > 0 and drowsy_mask.sum() > 0:
            epochs_alert = epochs[alert_mask]
            epochs_drowsy = epochs[drowsy_mask]
            
            ea = EuclideanAlignment()
            X_alert = epochs_alert.get_data(copy=True)
            X_drowsy = epochs_drowsy.get_data(copy=True)
            X_all = np.concatenate([X_alert, X_drowsy], axis=0)
            
            ea.fit(X_all, np.zeros(len(X_all)))
            X_alert_ea = ea.transform(X_alert, np.zeros(len(X_alert)))
            X_drowsy_ea = ea.transform(X_drowsy, np.zeros(len(X_drowsy)))
            
            all_alert_epochs.append(mne.EpochsArray(X_alert_ea, epochs_alert.info, verbose=False))
            all_drowsy_epochs.append(mne.EpochsArray(X_drowsy_ea, epochs_drowsy.info, verbose=False))
            
    if all_alert_epochs and all_drowsy_epochs:
        global_alert = mne.concatenate_epochs(all_alert_epochs, verbose=False)
        global_drowsy = mne.concatenate_epochs(all_drowsy_epochs, verbose=False)
        
        out_path = os.path.join(out_dir, "ea_topomap.png")
        plot_alert_vs_drowsy_topomaps(
            global_alert, 
            global_drowsy, 
            title="Euclidean Aligned Data (Global Average)", 
            out_path=out_path
        )
        print(f"Saved EA Topomap to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate advanced presentation figures.")
    parser.add_argument(
        "--figure", 
        type=str, 
        choices=["eegnet_weights", "ea_topomap", "adabn_tsne", "power_progression", "all"],
        default="all",
        help="Which figure to generate."
    )
    parser.add_argument("--out-dir", type=str, default="data/results/23_Subjects_Plots/Feature_Visualizations")
    parser.add_argument("--epochs-dir", type=str, default="data/SEED_VIG/epoched")
    parser.add_argument("--checkpoint-dir", type=str, default="data/results/experiments/23_Subjects_Stable/final_baseline_eegnet/checkpoints")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.figure in ["eegnet_weights", "all"]:
        plot_eegnet_spatial_filters(
            checkpoint_dir=args.checkpoint_dir,
            out_path=os.path.join(args.out_dir, "eegnet_spatial_filters.png")
        )
        
    if args.figure in ["ea_topomap", "all"]:
        generate_ea_topomaps(args.epochs_dir, args.out_dir)
        
    if args.figure in ["adabn_tsne", "all"]:
        ckpt_path = "data/results/experiments/23_Subjects_Stable/final_baseline_eegnet/checkpoints/eegnet_subject_10.pt"
        plot_adabn_tsne(
            ckpt_path=ckpt_path,
            target_subject="10",
            source_subjects=["11", "12"],
            epochs_dir=args.epochs_dir,
            out_path=os.path.join(args.out_dir, "adabn_tsne.png")
        )

    if args.figure in ["power_progression", "all"]:
        epochs_path = os.path.join(args.epochs_dir, "10_20151125_noon-epo.fif")
        base_csv = "data/results/experiments/23_Subjects_Stable/final_baseline_eegnet/predictions.csv"
        adabn_csv = "data/results/experiments/23_Subjects_Stable/final_eval_adabn_chronological_sweep/predictions.csv"
        ea_csv = "data/results/experiments/23_Subjects_Stable/final_eval_ea_chronological_sweep/predictions.csv"
        presentation_dir = args.out_dir
        
        plot_power_progression_over_time(
            epochs_path=epochs_path,
            baseline_csv_path=base_csv,
            adabn_csv_path=adabn_csv,
            ea_csv_path=ea_csv,
            out_path=os.path.join(args.out_dir, "biomarker_domain_adaptation.png")
        )

        from src.visualization.presentation_extras import (
            plot_average_prediction_overlay,
            plot_per_subject_adaptation_gain,
            plot_prediction_overlay,
            plot_all_subject_overlays,
            plot_global_prediction_distribution
        )
        
        print("Generating prediction overlays and presentation extras...")
        plot_prediction_overlay(base_csv, adabn_csv, ea_csv, os.path.join(presentation_dir, "prediction_overlay.png"))
        plot_average_prediction_overlay(base_csv, adabn_csv, ea_csv, os.path.join(presentation_dir, "average_prediction_overlay.png"))
        plot_all_subject_overlays(base_csv, adabn_csv, ea_csv, "data/results/Final_Plots_All/Subject_Overlays_Chronological")
        plot_global_prediction_distribution(base_csv, adabn_csv, ea_csv, os.path.join(presentation_dir, "global_prediction_distribution.png"))
        plot_per_subject_adaptation_gain(os.path.join(presentation_dir, "per_subject_adaptation_gain.png"))
        
        print("Generating timeline deepdive...")
        import subprocess
        subprocess.run(["python3", "scripts/plot_timeline.py"])
        
        print("\nAll presentation figures have been generated in:", presentation_dir)

if __name__ == "__main__":
    main()
