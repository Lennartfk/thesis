import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_stats(df, group_col):
    df['actual_minutes'] = df['n_target_adapt'] * 8.0 / 60.0
    agg = df.groupby(group_col).agg({
        'actual_minutes': 'mean',
        'balanced_accuracy': ['mean', 'sem']
    }).reset_index()
    agg.columns = [group_col, 'minutes', 'acc', 'acc_sem']
    return agg.sort_values('minutes')

def generate_combined_overview(out_path="data/results/Final_Plots_All/Overview/overview_combined_ea_adabn.pdf", include_title=True):
    base_dir = Path('data/results/experiments/23_Subjects_Stable')
    
    ea_cv = get_stats(pd.read_csv(base_dir / 'final_eval_ea_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
    ea_chrono = get_stats(pd.read_csv(base_dir / 'final_eval_ea_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
    ea_seq = get_stats(pd.read_csv(base_dir / 'eval_sequential_ea_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
    
    adabn_cv = get_stats(pd.read_csv(base_dir / 'final_eval_adabn_cv_sweep/fold_metrics.csv'), 'adaptation_target_fraction')
    adabn_chrono = get_stats(pd.read_csv(base_dir / 'final_eval_adabn_chronological_sweep/fold_metrics.csv'), 'chronological_minutes')
    adabn_seq = get_stats(pd.read_csv(base_dir / 'eval_sequential_adabn_sequential_window_sweep/fold_metrics.csv'), 'valid_minutes')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    plt.rcParams.update({'font.size': 14})
    
    def plot_on_ax(ax, cv_agg, chrono_agg, seq_agg, title):
        ax.plot(cv_agg['minutes'], cv_agg['acc'], label='Random Sampling (Idealized)', color='#009E73', linewidth=3, marker='o')
        ax.fill_between(cv_agg['minutes'], cv_agg['acc'] - cv_agg['acc_sem'], cv_agg['acc'] + cv_agg['acc_sem'], color='#009E73', alpha=0.15)
        
        ax.plot(chrono_agg['minutes'], chrono_agg['acc'], label='Chronological', color='#D55E00', linewidth=3, marker='s')
        ax.fill_between(chrono_agg['minutes'], chrono_agg['acc'] - chrono_agg['acc_sem'], chrono_agg['acc'] + chrono_agg['acc_sem'], color='#D55E00', alpha=0.15)
        
        ax.plot(seq_agg['minutes'], seq_agg['acc'], label='Sequential Block-wise', color='#0072B2', linewidth=3, marker='^')
        ax.fill_between(seq_agg['minutes'], seq_agg['acc'] - seq_agg['acc_sem'], seq_agg['acc'] + seq_agg['acc_sem'], color='#0072B2', alpha=0.15)
        
        if include_title:
            ax.set_title(title, weight='bold', fontsize=16)
        ax.set_xlabel("Mean Valid Calibration Data Used (Minutes)", weight='bold', fontsize=14)
        
        ax.set_xlim(0, 95)
        ax.set_ylim(0.65, 0.95)
        ax.grid(True, alpha=0.4, linestyle='--')

    plot_on_ax(ax1, ea_cv, ea_chrono, ea_seq, "Euclidean Alignment (EA)")
    ax1.set_ylabel("Mean Balanced Accuracy", weight='bold', fontsize=14)
    ax1.legend(loc='lower right', framealpha=0.9, title="Calibration Protocol", fontsize=12)

    plot_on_ax(ax2, adabn_cv, adabn_chrono, adabn_seq, "Adaptive Batch Normalization (AdaBN)")
    ax2.legend(loc='lower right', framealpha=0.9, title="Calibration Protocol", fontsize=12)
    
    if include_title:
        fig.suptitle("Comparison of Calibration Protocols: EA vs. AdaBN", weight='bold', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95] if include_title else [0, 0, 1, 1])
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_combined_overview()
