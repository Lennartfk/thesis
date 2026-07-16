import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_predictions(base_dir, baseline_dir, cutoff_minutes):
    base_dir = Path(base_dir)
    baseline_dir = Path(baseline_dir)
    
    dfs = []
    
    baseline_pred_file = baseline_dir / "predictions.csv"
    if baseline_pred_file.exists():
        df_base = pd.read_csv(baseline_pred_file)
        df_base["method"] = "EEGNet (Baseline)"
        df_base["epoch_minutes"] = df_base["epoch_index"] * 8 / 60.0
        if "target" in df_base.columns:
            df_base.rename(columns={"target": "y_true"}, inplace=True)
        df_base["correct"] = (df_base["y_true"] == df_base["y_pred"]).astype(float)
        dfs.append(df_base)
    else:
        print(f"Warning: Baseline predictions not found at {baseline_pred_file}")
        
    for method_dir in ["final_eval_ea_chronological_sweep", "final_eval_adabn_chronological_sweep"]:
        pred_file = base_dir / method_dir / "predictions.csv"
        if pred_file.exists():
            df_adapt = pd.read_csv(pred_file)
            df_adapt = df_adapt[np.isclose(df_adapt["chronological_minutes"], cutoff_minutes)].copy()
            if df_adapt.empty:
                print(f"Warning: No data found for cutoff {cutoff_minutes} in {method_dir}")
                continue
            
            method_name = "EEGNet + EA" if "ea" in method_dir else "EEGNet + AdaBN"
            df_adapt["method"] = method_name
            df_adapt["epoch_minutes"] = df_adapt["epoch_index"] * 8 / 60.0
            df_adapt["correct"] = (df_adapt["y_true"] == df_adapt["y_pred"]).astype(float)
            if "test_subject_id" in df_adapt.columns:
                df_adapt.rename(columns={"test_subject_id": "subject_id"}, inplace=True)
            dfs.append(df_adapt)
        else:
            print(f"Warning: Predictions not found at {pred_file}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def plot_accuracy_decay(df, cutoff_minutes, out_png, include_title=True):
    if df.empty:
        print("No data to plot.")
        return
        
    bin_size = 10
    df["time_bin"] = (df["epoch_minutes"] // bin_size) * bin_size + (bin_size / 2)
    
    
    from sklearn.metrics import balanced_accuracy_score
    def safe_bacc(g):
        try:
            return balanced_accuracy_score(g["y_true"], g["y_pred"])
        except:
            return np.nan

    subject_bacc = df.groupby(["method", "time_bin", "subject_id"]).apply(safe_bacc).reset_index(name="bacc")
    
    subject_bacc = subject_bacc.dropna()
    
    agg_df = subject_bacc.groupby(["method", "time_bin"])["bacc"].agg(["mean", "std", "count"]).reset_index()
    agg_df["se"] = agg_df["std"] / np.sqrt(agg_df["count"])
    
    agg_df = agg_df[agg_df["count"] >= 3]
    
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    
    palette = {
        "EEGNet (Baseline)": "#4C72B0",
        "EEGNet + EA": "#55A868",
        "EEGNet + AdaBN": "#C44E52"
    }
    
    for method in palette.keys():
        m_df = agg_df[agg_df["method"] == method].sort_values("time_bin")
        if m_df.empty:
            continue
        plt.plot(m_df["time_bin"], m_df["mean"], label=method, color=palette[method], lw=2.5, marker='o')
        plt.fill_between(m_df["time_bin"], m_df["mean"] - m_df["se"], m_df["mean"] + m_df["se"], color=palette[method], alpha=0.15)
        
    plt.axvline(x=cutoff_minutes, color='black', linestyle='--', lw=2, zorder=0)
    plt.text(cutoff_minutes - 2, 0.85, "Calibration Period\n(Data used for Adaptation)", ha='right', va='center', rotation=90, fontsize=11)
    plt.text(cutoff_minutes + 2, 0.85, "Evaluation Period\n(Adapted Model Predicting)", ha='left', va='center', rotation=90, fontsize=11)
    
    if include_title:
        plt.title(f"Balanced Accuracy Decay Over Time (Adapted on first {cutoff_minutes} mins)", fontsize=16, pad=15)
    plt.xlabel("Evaluation Timeline (Minutes into driving session)", fontsize=14)
    plt.ylabel("Balanced Accuracy (Mean ± SEM)", fontsize=14)
    
    y_min = agg_df["mean"].min() - agg_df["se"].max()
    y_max = agg_df["mean"].max() + agg_df["se"].max()
    y_range = y_max - y_min
    plt.ylim(max(0.0, y_min - 0.05), min(1.0, y_max + y_range * 0.3))
    
    plt.legend(loc="upper right", fontsize=12)
    
    plt.xlim(0, 125)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Successfully generated decay plot at {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/results/experiments", help="experiments base dir")
    parser.add_argument("--baseline", default="data/results/experiments/Baselines/Deep/Cross-subject/EEGNET_Baseline", help="baseline dir")
    parser.add_argument("--cutoff", type=int, default=10, help="Chronological cutoff in minutes to plot")
    parser.add_argument("--out-png", default="data/results/presentation_new/chronological_decay_plot.png")
    args = parser.parse_args()
    
    df = load_predictions(args.base, args.baseline, args.cutoff)
    plot_accuracy_decay(df, args.cutoff, args.out_png)

if __name__ == "__main__":
    main()
