#!/usr/bin/env python3
"""Analyze adaptation fraction sweeps and test for statistical significance."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pingouin as pg


def load_baseline(baseline_dir):
    metric_file = os.path.join(baseline_dir, "fold_metrics.csv")
    if not os.path.exists(metric_file):
        raise FileNotFoundError(f"Baseline file not found: {metric_file}")
    df = pd.read_csv(metric_file)
    df["test_subject_id"] = df["test_subject_id"].astype(str)
    return df[["test_subject_id", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "baseline_accuracy"}
    )


def load_adaptation_sweeps(base_dir, sweep_type="fraction"):
    rows = []
    
    if sweep_type == "fraction":
        sweep_dirs = ["final_eval_adabn_cv_sweep", "final_eval_ea_cv_sweep"]
        frac_col = "adaptation_target_fraction"
    elif sweep_type == "chronological":
        # Note: the sweep dir is 'chronological' but the metric inside might be tracked as fraction or time
        sweep_dirs = ["final_eval_adabn_chronological_sweep", "final_eval_ea_chronological_sweep"]
        frac_col = "chronological_minutes"
    elif sweep_type == "fixed_window":
        # Fixed window has fraction column tracking window size
        sweep_dirs = ["final_eval_adabn_fixed_window_sweep", "final_eval_ea_fixed_window_sweep"]
        frac_col = "valid_minutes"
    elif sweep_type == "sequential_window":
        sweep_dirs = ["eval_sequential_adabn_sequential_window_sweep", "eval_sequential_ea_sequential_window_sweep"]
        frac_col = "valid_minutes"
        
    for d in sweep_dirs:
        metric_file = os.path.join(base_dir, d, "fold_metrics.csv")
        if not os.path.exists(metric_file):
            continue
            
        df = pd.read_csv(metric_file)
        for _, row in df.iterrows():
            if frac_col not in row:
                continue
            rows.append({
                "test_subject_id": str(row["test_subject_id"]),
                "fraction": float(row[frac_col]),
                "balanced_accuracy": float(row.get("balanced_accuracy", np.nan)),
                "method": str(row["adaptation_name"])
            })
    return pd.DataFrame(rows)


def run_statistical_tests(df_adapt, df_base):
    # Merge adaptation data with baseline data on subject ID
    df = pd.merge(df_adapt, df_base, on="test_subject_id")
    
    results = []
    methods = df["method"].unique()
    fractions = sorted(df["fraction"].unique())
    
    for method in methods:
        for fraction in fractions:
            sub = df[(df["method"] == method) & (df["fraction"] == fraction)]
            if len(sub) == 0:
                continue
            
            # Pingouin paired t-test: alternative='greater' tests if adapt > baseline
            pt = pg.ttest(
                sub["balanced_accuracy"], 
                sub["baseline_accuracy"],
                paired=True,
                alternative="greater"
            )
            
            mean_adapt = sub["balanced_accuracy"].mean()
            std_adapt = sub["balanced_accuracy"].std()
            se_adapt = std_adapt / np.sqrt(len(sub))
            
            results.append({
                "method": method,
                "fraction": fraction,
                "mean_acc": mean_adapt,
                "se_acc": se_adapt,
                "t_stat": pt["T"].iloc[0],
                "p_value": pt["p_val"].iloc[0],
                "cohen_d": pt["cohen_d"].iloc[0],
                "n_subjects": len(sub)
            })
            
    res_df = pd.DataFrame(results)
    
    # Benjamini-Hochberg FDR correction using Pingouin
    if len(res_df) > 0:
        reject, pvals_corrected = pg.multicomp(res_df["p_value"].values, method='fdr_bh')
        res_df["p_val_corrected"] = pvals_corrected
        res_df["significant"] = reject
        
    return res_df


def plot_results(stats_df, baseline_mean, baseline_se, out_png, sweep_type="fraction", error_bars=False, in_minutes=False):
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    plt.figure(figsize=(8, 5))
    
    # Plot baseline
    plt.axhline(baseline_mean, color="black", linestyle="--", linewidth=2, label="Baseline EEGNet")
    
    methods = sorted(stats_df["method"].unique())
    markers = {"ea": "o", "adabn": "s", "ea_adabn": "^"}
    colors = {"ea": "#55A868", "adabn": "#C44E52", "ea_adabn": "#4C72B0"}
    labels = {"ea": "EA", "adabn": "AdaBN", "ea_adabn": "EA + AdaBN"}
    
    for method in methods:
        sub = stats_df[stats_df["method"] == method].sort_values("fraction")
        x = sub["fraction"].values
        # Only scale for original fraction plot
        if sweep_type == "fraction" and in_minutes:
            x = x * 72.724
        y = sub["mean_acc"].values
        yerr = sub["se_acc"].values
        
        c = colors.get(method, "#333333")
        m = markers.get(method, "o")
        lbl = labels.get(method, method)
        
        plt.plot(x, y, marker=m, color=c, label=lbl, linewidth=2.5, markersize=7)
        if error_bars:
            plt.fill_between(x, y - yerr, y + yerr, color=c, alpha=0.15)
        
        # Mark significant points
        sig = sub[sub["significant"]]
        if not sig.empty:
            sig_x = sig["fraction"].values
            if sweep_type == "fraction" and in_minutes:
                sig_x = sig_x * 72.724
            plt.scatter(
                sig_x, sig["mean_acc"] + 0.015, 
                marker="*", color=c, s=150, zorder=5
            )

    # Plot dummy scatter for legend of significance
    plt.scatter([], [], marker="*", color="black", s=150, label="Significant vs Baseline\n(p < 0.05 FDR)")
    
    if sweep_type == "fraction":
        xlabel = "Target Subject Data for Adaptation (Minutes)" if in_minutes else "Fraction of Target Subject Data for Adaptation"
    elif sweep_type == "chronological":
        xlabel = "Absolute Chronological Time of Drive (Minutes)"
    elif sweep_type == "fixed_window":
        xlabel = "Target Subject Valid Data (Minutes)"
    elif sweep_type == "sequential_window":
        xlabel = "Sequential Window Calibration Duration (Minutes)"
        
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Mean Balanced Accuracy", fontweight="bold")
    plt.title("Domain Adaptation Gains", pad=15, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/results/experiments", help="experiments base dir")
    p.add_argument("--baseline", default="data/results/experiments/Baselines/Deep/Cross-subject/EEGNET_Baseline", help="baseline dir")
    p.add_argument("--out-csv", default="data/results/presentation/adaptation_fraction_stats.csv")
    p.add_argument("--out-png", default="data/results/presentation/adaptation_fraction_sweep.png")
    p.add_argument("--sweep-type", choices=["fraction", "chronological", "fixed_window", "sequential_window"], default="fraction")
    args = p.parse_args()

    print("Loading baseline...")
    df_base = load_baseline(args.baseline)
    baseline_mean = df_base["baseline_accuracy"].mean()
    baseline_se = df_base["baseline_accuracy"].std() / np.sqrt(len(df_base))

    print("Loading adaptation sweeps...")
    df_adapt = load_adaptation_sweeps(args.base, sweep_type=args.sweep_type)
    
    if df_adapt.empty:
        print(f"No sweep data found for sweep_type={args.sweep_type}! Run the SLURM sweeps first.")
        return

    print("Running statistical tests...")
    stats_df = run_statistical_tests(df_adapt, df_base)
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    stats_df.to_csv(args.out_csv, index=False)
    print(f"Saved stats to {args.out_csv}")

    print("Plotting results...")
    plot_results(stats_df, baseline_mean, baseline_se, args.out_png, sweep_type=args.sweep_type, error_bars=False)
    print(f"Saved plot to {args.out_png}")
    
    out_png_err = args.out_png.replace(".png", "_errorbars.png")
    plot_results(stats_df, baseline_mean, baseline_se, out_png_err, sweep_type=args.sweep_type, error_bars=True)
    print(f"Saved plot with error bars to {out_png_err}")
    
    if args.sweep_type == "fraction":
        out_png_min = args.out_png.replace(".png", "_minutes.png")
        plot_results(stats_df, baseline_mean, baseline_se, out_png_min, sweep_type=args.sweep_type, error_bars=True, in_minutes=True)
        print(f"Saved plot in minutes to {out_png_min}")


if __name__ == "__main__":
    main()
