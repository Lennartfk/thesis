import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.metrics import balanced_accuracy_score

def calculate_bacc(group):
    try:
        if len(np.unique(group["target" if "target" in group.columns else "y_true"])) == 1:
            y_t = group["target" if "target" in group.columns else "y_true"]
            return (y_t == group["y_pred"]).mean()
        y_t = group["target" if "target" in group.columns else "y_true"]
        return balanced_accuracy_score(y_t, group["y_pred"])
    except:
        return np.nan

def generate_heatmap(base_dir, sweep_type, out_png, title, y_label, method="ea", include_title=True):
    base_dir = Path(base_dir)
    
    if sweep_type == "chronological":
        method_dir = f"final_eval_{method}_chronological_sweep"
        y_col = "chronological_minutes"
    elif sweep_type == "fixed_window":
        method_dir = f"final_eval_{method}_fixed_window_sweep"
        y_col = "fixed_window_minutes"
    else:
        return

    pred_file = base_dir / method_dir / "predictions.csv"
    if not pred_file.exists():
        print(f"File not found: {pred_file}")
        return
        
    df = pd.read_csv(pred_file)
    df["epoch_minutes"] = df["epoch_index"] * 8 / 60.0
    
    bin_size = 10
    df["time_bin"] = (df["epoch_minutes"] // bin_size) * bin_size + (bin_size / 2)
    
    agg_df = df.groupby([y_col, "time_bin"]).apply(calculate_bacc, include_groups=False).reset_index()
    agg_df.columns = [y_col, "time_bin", "balanced_accuracy"]
    
    if sweep_type == "chronological":
        agg_df = agg_df[agg_df["time_bin"] > agg_df[y_col]]
        
    pivot_df = agg_df.pivot(index=y_col, columns="time_bin", values="balanced_accuracy")
    pivot_df = pivot_df.sort_index(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white", font_scale=1.2)
    
    ax = sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Balanced Accuracy'},
                     linewidths=.5, vmin=0.5, vmax=0.9)
                     
    if include_title:
        ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Evaluation Time Since Start of Drive (Minutes)", fontsize=14, labelpad=10)
    ax.set_ylabel(y_label, fontsize=14, labelpad=10)
    
    ax.set_xticklabels([f"{int(x-5)}-{int(x+5)}m" for x in pivot_df.columns], rotation=45)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", format="pdf")
    plt.close()

def generate_heatmap_gain(base_dir, baseline_dir, out_png, method="ea", include_title=True):
    base_dir = Path(base_dir)
    baseline_dir = Path(baseline_dir)
    
    baseline_pred_file = baseline_dir / "predictions.csv"
    if not baseline_pred_file.exists():
        print("No baseline file")
        return
        
    df_base = pd.read_csv(baseline_pred_file)
    df_base["epoch_minutes"] = df_base["epoch_index"] * 8 / 60.0
    bin_size = 10
    df_base["time_bin"] = (df_base["epoch_minutes"] // bin_size) * bin_size + (bin_size / 2)
    
    base_bacc = df_base.groupby("time_bin").apply(calculate_bacc, include_groups=False).reset_index()
    base_bacc.columns = ["time_bin", "base_bacc"]
    
    method_dir = f"final_eval_{method}_chronological_sweep"
    pred_file = base_dir / method_dir / "predictions.csv"
    
    df_ea = pd.read_csv(pred_file)
    df_ea["epoch_minutes"] = df_ea["epoch_index"] * 8 / 60.0
    df_ea["time_bin"] = (df_ea["epoch_minutes"] // bin_size) * bin_size + (bin_size / 2)
    
    ea_bacc = df_ea.groupby(["chronological_minutes", "time_bin"]).apply(calculate_bacc, include_groups=False).reset_index()
    ea_bacc.columns = ["chronological_minutes", "time_bin", f"{method}_bacc"]
    
    merged = pd.merge(ea_bacc, base_bacc, on="time_bin", how="inner")
    merged["gain"] = merged[f"{method}_bacc"] - merged["base_bacc"]
    
    merged = merged[merged["time_bin"] > merged["chronological_minutes"]]
    
    pivot_df = merged.pivot(index="chronological_minutes", columns="time_bin", values="gain")
    pivot_df = pivot_df.sort_index(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white", font_scale=1.2)
    
    ax = sns.heatmap(pivot_df, cmap="vlag", annot=True, fmt="+.2f", center=0,
                     cbar_kws={'label': f'Gain in Balanced Accuracy ({method.upper()} - Baseline)'},
                     linewidths=.5)
                     
    if include_title:
        ax.set_title(f"Adaptation Gain over Baseline (Chronological) - {method.upper()}", fontsize=16, pad=20)
    ax.set_xlabel("Evaluation Time Since Start of Drive (Minutes)", fontsize=14, labelpad=10)
    ax.set_ylabel("Calibration Period (Minutes)", fontsize=14, labelpad=10)
    
    ax.set_xticklabels([f"{int(x-5)}-{int(x+5)}m" for x in pivot_df.columns], rotation=45)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", format="pdf")
    plt.close()
