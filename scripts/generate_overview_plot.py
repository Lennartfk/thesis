import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def get_stats(df, group_col):
    df['actual_minutes'] = df['n_target_adapt'] * 8.0 / 60.0
    agg = df.groupby(group_col).agg({
        'actual_minutes': 'mean',
        'balanced_accuracy': ['mean', 'sem']
    }).reset_index()
    agg.columns = [group_col, 'minutes', 'acc', 'acc_sem']
    return agg.sort_values('minutes')

def plot_overview(cv_agg, chrono_agg, seq_agg, title, out_path, include_title=True):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    
    # Random Sampling (CV)
    plt.plot(cv_agg['minutes'], cv_agg['acc'], label='Random Sampling (Idealized)', color='#009E73', linewidth=3, marker='o')
    plt.fill_between(cv_agg['minutes'], cv_agg['acc'] - cv_agg['acc_sem'], cv_agg['acc'] + cv_agg['acc_sem'], color='#009E73', alpha=0.15)
    
    # Chronological
    plt.plot(chrono_agg['minutes'], chrono_agg['acc'], label='Chronological', color='#D55E00', linewidth=3, marker='s')
    plt.fill_between(chrono_agg['minutes'], chrono_agg['acc'] - chrono_agg['acc_sem'], chrono_agg['acc'] + chrono_agg['acc_sem'], color='#D55E00', alpha=0.15)
    
    # Sequential Window
    plt.plot(seq_agg['minutes'], seq_agg['acc'], label='Sequential Block-wise', color='#0072B2', linewidth=3, marker='^')
    plt.fill_between(seq_agg['minutes'], seq_agg['acc'] - seq_agg['acc_sem'], seq_agg['acc'] + seq_agg['acc_sem'], color='#0072B2', alpha=0.15)
    
    if include_title:
        plt.title(title, weight='bold')
    plt.xlabel("Mean Valid Calibration Data Used (Minutes)", weight='bold')
    plt.ylabel("Mean Balanced Accuracy", weight='bold')
    
    plt.xlim(0, 95)
    plt.ylim(0.65, 0.95)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(loc='lower right', framealpha=0.9, title="Calibration Protocol")
    
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"Saved {out_path}")

def plot_comparison(ea_seq, adabn_seq, baseline_acc, out_path, include_title=True):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    
    # EA Sequential
    plt.plot(ea_seq['minutes'], ea_seq['acc'], label='Euclidean Alignment', color='#0072B2', linewidth=3, marker='o')
    plt.fill_between(ea_seq['minutes'], ea_seq['acc'] - ea_seq['acc_sem'], ea_seq['acc'] + ea_seq['acc_sem'], color='#0072B2', alpha=0.15)
    
    # AdaBN Sequential
    plt.plot(adabn_seq['minutes'], adabn_seq['acc'], label='Adaptive Batch Normalization', color='#D55E00', linewidth=3, marker='s')
    plt.fill_between(adabn_seq['minutes'], adabn_seq['acc'] - adabn_seq['acc_sem'], adabn_seq['acc'] + adabn_seq['acc_sem'], color='#D55E00', alpha=0.15)
    
    # Baseline
    plt.axhline(y=baseline_acc, color='#000000', linestyle='--', linewidth=3, label='Uncalibrated Baseline')
    
    if include_title:
        plt.title("Direct Comparison: EA vs AdaBN (Sequential Block-wise)", weight='bold')
    plt.xlabel("Mean Valid Calibration Data Used (Minutes)", weight='bold')
    plt.ylabel("Mean Balanced Accuracy", weight='bold')
    
    plt.xlim(0, 95)
    plt.ylim(0.65, 0.95)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(loc='lower right', framealpha=0.9, title="Method")
    
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"Saved {out_path}")

def generate_all():
    base_dir = Path('data/results/experiments/23_Subjects_Stable')
    out_dir = Path("data/results/Final_Plots_All/Overview")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EA
    ea_cv = get_stats(pd.read_csv(base_dir / 'final_eval_ea_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
    ea_chrono = get_stats(pd.read_csv(base_dir / 'final_eval_ea_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
    ea_seq = get_stats(pd.read_csv(base_dir / 'eval_sequential_ea_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
    
    # Load AdaBN
    adabn_cv = get_stats(pd.read_csv(base_dir / 'final_eval_adabn_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
    adabn_chrono = get_stats(pd.read_csv(base_dir / 'final_eval_adabn_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
    adabn_seq = get_stats(pd.read_csv(base_dir / 'eval_sequential_adabn_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
    
    # Baseline
    # We can calculate baseline accuracy from baseline predictions
    from sklearn.metrics import balanced_accuracy_score
    import numpy as np
    base_df = pd.read_csv(base_dir / 'final_baseline_eegnet/predictions.csv')
    subjects = base_df['subject_id'].unique()
    sub_accs = []
    for s in subjects:
        sdf = base_df[base_df['subject_id'] == s]
        sub_accs.append(balanced_accuracy_score(sdf['target'], sdf['y_pred']))
    baseline_acc = np.mean(sub_accs)
    
    # Generate Plots
    plot_overview(ea_cv, ea_chrono, ea_seq, "Overview: Calibration Methods Comparison (Euclidean Alignment)", out_dir / "overview_ea.png")
    plot_overview(adabn_cv, adabn_chrono, adabn_seq, "Overview: Calibration Methods Comparison (AdaBN)", out_dir / "overview_adabn.png")
    plot_comparison(ea_seq, adabn_seq, baseline_acc, out_dir / "overview_comparison_sequential.png")

if __name__ == "__main__":
    generate_all()
