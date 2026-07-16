#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.plot_3d_surface import plot_3d
from scripts.analyze_subject_perclos_stats import analyze_stats
from scripts.plot_chronological_decay import load_predictions, plot_accuracy_decay
from scripts.generate_final_presentation_plots import plot_sweep, plot_adaptation_gain
from scripts.stats_adaptation_fraction import load_adaptation_sweeps, load_baseline
from scripts.generate_ea_scatter_plot import generate_scatter_plot
from scripts.plot_heatmaps import generate_heatmap, generate_heatmap_gain
from scripts.generate_overview_plot import get_stats, plot_overview, plot_comparison
from scripts.generate_combined_overview import generate_combined_overview
from scripts.generate_random_10pct_overlays import generate_overlays
import pandas as pd

def main():
    base_results = Path("data/results/experiments/23_Subjects_Stable")
    thesis_root = Path("data/results/Thesis_Figures")
    
    for include_title in [True, False]:
        if include_title:
            out_root = thesis_root / "figures_with_titles"
            title_flag_str = ""
            print("=====================================")
            print("GENERATING FIGURES WITH TITLES")
            print("=====================================")
        else:
            out_root = thesis_root / "figures"
            title_flag_str = "--no-title"
            print("=====================================")
            print("GENERATING FIGURES WITHOUT TITLES")
            print("=====================================")
            
        # 1. 3D_Plots
        d_3d = out_root / "3D_Plots"
        os.makedirs(d_3d, exist_ok=True)
        for method in ["ea", "adabn"]:
            out_path = d_3d / f"3D_Chronological_Accuracy_Surface_{method.upper()}.pdf"
            plot_3d(base_results, method=method, include_title=include_title, out_path=str(out_path))

        # 2. Dataset_Stats
        d_stats = out_root / "Dataset_Stats"
        analyze_stats(out_dir=str(d_stats), include_title=include_title)

        # 3. Decay_Over_Time
        d_decay = out_root / "Decay_Over_Time"
        os.makedirs(d_decay, exist_ok=True)
        for cutoff in [10, 30, 60]:
            df_decay = load_predictions(base_results, base_results / "final_baseline_eegnet", cutoff)
            out_png = d_decay / f"Chronological_Decay_Cutoff_{cutoff}m.pdf"
            plot_accuracy_decay(df_decay, cutoff, str(out_png), include_title=include_title)

        # 4 & 10. Domain_Adaptation_Gain & Sweep_Summaries_1D
        d_gain = out_root / "Domain_Adaptation_Gain"
        d_sweep = out_root / "Sweep_Summaries_1D"
        os.makedirs(d_gain, exist_ok=True)
        os.makedirs(d_sweep, exist_ok=True)
        
        df_base = load_baseline(base_results / "final_baseline_eegnet")
        sweeps = [
            ("fraction", "adaptation_fraction_sweep.pdf"),
            ("chronological", "chronological_sweep.pdf"),
            ("fixed_window", "fixed_window_sweep.pdf"),
            ("sequential_window", "sequential_window_sweep.pdf")
        ]
        
        for stype, fname in sweeps:
            df_adapt = load_adaptation_sweeps(base_results, sweep_type=stype)
            if not df_adapt.empty:
                plot_sweep(df_adapt, df_base, stype, str(d_sweep / fname), include_title=include_title)
                plot_adaptation_gain(df_adapt, df_base, str(d_gain / f"gain_bar_{stype}.pdf"), include_title=include_title)

        # 5. Feature_Visualizations
        d_feat = out_root / "Feature_Visualizations"
        generate_scatter_plot(out_path=str(d_feat / "ea_scatter_real_data_2d.pdf"), include_title=include_title)

        # 6. Heatmaps
        d_heat = out_root / "Heatmaps"
        os.makedirs(d_heat, exist_ok=True)
        for method in ["ea", "adabn"]:
            generate_heatmap(
                base_results, 
                "chronological", 
                str(d_heat / f"Chronological_Accuracy_Heatmap_{method.upper()}.pdf"),
                f"Chronological Sweep: Balanced Accuracy over Time ({method.upper()})",
                "Calibration Period (Minutes)",
                method=method,
                include_title=include_title
            )
            generate_heatmap_gain(
                base_results, 
                base_results / "final_baseline_eegnet",
                str(d_heat / f"Chronological_Gain_Heatmap_{method.upper()}.pdf"),
                method=method,
                include_title=include_title
            )

        # 7. Overview
        d_over = out_root / "Overview"
        os.makedirs(d_over, exist_ok=True)
        
        # We need the CV, Chrono, and Seq aggregations
        # EA
        ea_cv = get_stats(pd.read_csv(base_results / 'final_eval_ea_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
        ea_chrono = get_stats(pd.read_csv(base_results / 'final_eval_ea_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
        ea_seq = get_stats(pd.read_csv(base_results / 'eval_sequential_ea_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
        # AdaBN
        adabn_cv = get_stats(pd.read_csv(base_results / 'final_eval_adabn_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
        adabn_chrono = get_stats(pd.read_csv(base_results / 'final_eval_adabn_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
        adabn_seq = get_stats(pd.read_csv(base_results / 'eval_sequential_adabn_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
        # Baseline
        base_df = pd.read_csv(base_results / 'final_baseline_eegnet/predictions.csv')
        from sklearn.metrics import balanced_accuracy_score
        sub_accs = [balanced_accuracy_score(sdf['target'], sdf['y_pred']) for s, sdf in base_df.groupby('subject_id')]
        baseline_acc = sum(sub_accs) / len(sub_accs)

        plot_overview(ea_cv, ea_chrono, ea_seq, "Overview: Calibration Methods Comparison (Euclidean Alignment)", 
                      str(d_over / "overview_ea.pdf"), include_title=include_title)
        plot_overview(adabn_cv, adabn_chrono, adabn_seq, "Overview: Calibration Methods Comparison (AdaBN)", 
                      str(d_over / "overview_adabn.pdf"), include_title=include_title)
        plot_comparison(ea_seq, adabn_seq, baseline_acc, str(d_over / "overview_comparison_sequential.pdf"), include_title=include_title)
        generate_combined_overview(out_path=str(d_over / "overview_combined_ea_adabn.pdf"), include_title=include_title)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        # 8. Subject_Overlays
        d_deep = out_root / "Subject_Overlays"
        os.makedirs(d_deep, exist_ok=True)
        cmd_timeline = ["python3", "scripts/plot_timeline.py", "--out-path", str(d_deep / "prediction_timeline_deepdive.pdf")]
        if not include_title:
            cmd_timeline.append("--no-title")
        subprocess.run(cmd_timeline, env=env, check=True)

        # 9. Subject_Overlays_Random_10pct
        d_rand = out_root / "Subject_Overlays_Random_10pct"
        generate_overlays(out_dir=str(d_rand), include_title=include_title)

        # 11. Topomaps
        d_topo = out_root / "Topomaps"
        cmd_topo = ["python3", "scripts/generate_presentation_topomaps.py", "--out-dir", str(d_topo)]
        if not include_title:
            cmd_topo.append("--no-title")
        subprocess.run(cmd_topo, env=env, check=True)

    print("All sets generated successfully!")

if __name__ == "__main__":
    main()
