import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(".")
from scripts.plot_heatmaps import generate_heatmap, generate_heatmap_gain
from scripts.plot_3d_surface import plot_3d
from scripts.plot_chronological_decay import load_predictions, plot_accuracy_decay
from scripts.stats_adaptation_fraction import load_adaptation_sweeps, load_baseline

def plot_sweep(df_adapt, df_base, sweep_type, out_png, include_title=True):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    
    if sweep_type == "fraction":
        x_col = "fraction"
        xlabel = "Adaptation Data Fraction"
    elif sweep_type == "chronological":
        x_col = "fraction" # Note: load_adaptation_sweeps renames the sweep column to 'fraction' for all of them!
        xlabel = "Calibration Period (Minutes)"
    elif sweep_type == "fixed_window":
        x_col = "fraction"
        xlabel = "Sliding Window Size (Minutes)"
    elif sweep_type == "sequential_window":
        x_col = "fraction"
        x_col = "fraction"
        xlabel = "Sequential Calibration Window (Minutes)"
        
    palette = {"ea": "#0072B2", "adabn": "#D55E00"} # Okabe-Ito (Blue, Vermilion)
    markers = {"ea": "o", "adabn": "s"}
    
    base_mean = df_base["baseline_accuracy"].mean()
    base_se = df_base["baseline_accuracy"].std() / np.sqrt(len(df_base))
    plt.axhline(base_mean, color="#000000", linestyle="--", label="Uncalibrated Baseline", linewidth=3)
    plt.fill_between(sorted(df_adapt[x_col].unique()), 
                     base_mean - base_se, base_mean + base_se, 
                     color="#000000", alpha=0.1)
                     
    if sweep_type == "fraction":
        df_adapt = df_adapt[df_adapt[x_col] < 1.0]
    
    agg = df_adapt.groupby([x_col, "method"])["balanced_accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    
    for method in agg["method"].unique():
        m_df = agg[agg["method"] == method].sort_values(x_col)
        name = "Euclidean Alignment" if method == "ea" else "Adaptive Batch Normalization"
        color = palette.get(method, "black")
        marker = markers.get(method, "o")
        
        plt.plot(m_df[x_col], m_df["mean"], label=name, color=color, marker=marker, linewidth=3)
        plt.fill_between(m_df[x_col], 
                         m_df["mean"] - m_df["se"], 
                         m_df["mean"] + m_df["se"], 
                         color=color, alpha=0.15)
                     
    if include_title:
        plt.title(f"{sweep_type.replace('_', ' ').title()} Sweep", weight='bold')
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel("Mean Balanced Accuracy", weight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    
    plt.ylim(0.65, 0.95)
    
    if sweep_type == "fraction":
        plt.xlim(0.0, 0.95)
    else:
        plt.xlim(0, 95)
    
    
    plt.legend(loc="upper right")
    plt.savefig(out_png, bbox_inches="tight", format="pdf")
    plt.close()

def plot_adaptation_gain(df_adapt, df_base, out_png, include_title=True):
    max_frac = df_adapt["fraction"].max()
    df_best = df_adapt[np.isclose(df_adapt["fraction"], max_frac)]
    
    df_merged = pd.merge(df_best, df_base, on="test_subject_id")
    df_merged["gain"] = df_merged["balanced_accuracy"] - df_merged["baseline_accuracy"]
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    palette = {"ea": "#55A868", "adabn": "#C44E52"}
    
    sns.barplot(data=df_merged, x="test_subject_id", y="gain", hue="method", palette=palette)
    plt.axhline(0, color="black", lw=1)
    
    if include_title:
        plt.title(f"Per-Subject Gain in Balanced Accuracy (Max Setting)", fontsize=16)
    plt.xlabel("Subject ID", fontsize=14)
    plt.ylabel("Gain over Baseline", fontsize=14)
    plt.legend(title="Method")
    plt.savefig(out_png, bbox_inches="tight", format="pdf")
    plt.close()

def main():
    base_out = Path("data/results/Final_Plots_All")
    
    dirs = [
        base_out / "Heatmaps",
        base_out / "3D_Plots",
        base_out / "Decay_Over_Time",
        base_out / "Sweep_Summaries_1D",
        base_out / "Domain_Adaptation_Gain"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    print("1. Generating Heatmaps...")
    for method in ["ea", "adabn"]:
        generate_heatmap(
            "data/results/experiments/23_Subjects_Stable", 
            "chronological", 
            str(base_out / "Heatmaps" / f"Chronological_Accuracy_Heatmap_{method.upper()}.png"),
            f"Chronological Sweep: Balanced Accuracy over Time ({method.upper()})",
            "Calibration Period (Minutes)",
            method=method
        )
        generate_heatmap_gain(
            "data/results/experiments/23_Subjects_Stable", 
            "data/results/experiments/23_Subjects_Stable/final_baseline_eegnet",
            str(base_out / "Heatmaps" / f"Chronological_Gain_Heatmap_{method.upper()}.png"),
            method=method
        )
        
    print("2. Generating 3D Surface Plots...")
    for method in ["ea", "adabn"]:
        plot_3d("data/results/experiments/23_Subjects_Stable", method=method)
        src_3d = "data/results/presentation_new/3d_accuracy_surface.png"
        if os.path.exists(src_3d):
            os.rename(src_3d, base_out / "3D_Plots" / f"3D_Chronological_Accuracy_Surface_{method.upper()}.png")
        
    print("3. Generating Decay Over Time Plots...")
    for cutoff in [10, 30, 60]:
        df_decay = load_predictions("data/results/experiments/23_Subjects_Stable", "data/results/experiments/23_Subjects_Stable/final_baseline_eegnet", cutoff)
        out_png = base_out / "Decay_Over_Time" / f"Chronological_Decay_Cutoff_{cutoff}m.png"
        plot_accuracy_decay(df_decay, cutoff, str(out_png))
        
    print("4. Generating 1D Sweep Summaries and Adaptation Gain...")
    baseline_dir = "data/results/experiments/23_Subjects_Stable/final_baseline_eegnet"
    df_base = load_baseline(baseline_dir)
    
    sweeps = [
        ("fraction", "adaptation_fraction_sweep.png"),
        ("chronological", "chronological_sweep.png"),
        ("fixed_window", "fixed_window_sweep.png"),
        ("sequential_window", "sequential_window_sweep.png")
    ]
    
    for stype, fname in sweeps:
        df_adapt = load_adaptation_sweeps("data/results/experiments/23_Subjects_Stable", sweep_type=stype)
        if df_adapt.empty:
            print(f"No data for {stype}")
            continue
            
        out_png = base_out / "Sweep_Summaries_1D" / fname
        plot_sweep(df_adapt, df_base, stype, str(out_png))
        
        gain_png = base_out / "Domain_Adaptation_Gain" / f"gain_bar_{stype}.png"
        plot_adaptation_gain(df_adapt, df_base, str(gain_png))

    print("All plots generated and organized into data/results/Final_Plots_All!")

if __name__ == "__main__":
    main()
