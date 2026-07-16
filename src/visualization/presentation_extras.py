import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def smooth(y, window_size):
    """Applies a moving average to smooth out the signal."""
    box = np.ones(window_size)/window_size
    y_smooth = np.convolve(y, box, mode='valid')
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    return np.pad(y_smooth, (pad_left, pad_right), mode='constant', constant_values=np.nan)

def load_df(path):
    df = pd.read_csv(path)
    if 'test_subject_id' in df.columns: 
        df = df.rename(columns={'test_subject_id': 'subject_id'})
    if 'chronological_minutes' in df.columns: 
        df = df[df['chronological_minutes'] == 10.0]
    return df

def plot_prediction_overlay(baseline_csv_path: str, adabn_csv_path: str, ea_csv_path: str, out_path: str):
    """Plot Baseline vs AdaBN predictions overlaying true PERCLOS for Subject 2."""
    subject_id = 2
    window_size = 50
    
    baseline_df = load_df(baseline_csv_path)
    baseline_df = baseline_df[baseline_df['subject_id'] == subject_id].sort_values('epoch_index')
    
    adabn_df = load_df(adabn_csv_path)
    adabn_df = adabn_df[adabn_df['subject_id'] == subject_id].sort_values('epoch_index')
    
    ea_df = load_df(ea_csv_path)
    ea_df = ea_df[ea_df['subject_id'] == subject_id].sort_values('epoch_index')
    
    merged = baseline_df[['epoch_index', 'perclos', 'y_score']].rename(columns={'y_score': 'base_pred'})
    merged = merged.merge(adabn_df[['epoch_index', 'y_score']].rename(columns={'y_score': 'adabn_pred'}), on='epoch_index', how='left')
    merged = merged.merge(ea_df[['epoch_index', 'y_score']].rename(columns={'y_score': 'ea_pred'}), on='epoch_index', how='left')
    
    time_axis = merged['epoch_index'].values
    true_perclos = smooth(merged['perclos'].values, window_size)
    base_pred = smooth(merged['base_pred'].values, window_size)
    adabn_pred = smooth(merged['adabn_pred'].values, window_size)
    ea_pred = smooth(merged['ea_pred'].values, window_size)
    
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(time_axis, true_perclos, color='#2ca02c', linewidth=4, label='True PERCLOS (Fatigue)', zorder=1)
    
    v = ~np.isnan(true_perclos) & ~np.isnan(base_pred)
    b_rmse = np.sqrt(np.mean((true_perclos[v] - base_pred[v])**2)) if sum(v)>0 else 0
    va = ~np.isnan(true_perclos) & ~np.isnan(adabn_pred)
    a_rmse = np.sqrt(np.mean((true_perclos[va] - adabn_pred[va])**2)) if sum(va)>0 else 0
    ve = ~np.isnan(true_perclos) & ~np.isnan(ea_pred)
    e_rmse = np.sqrt(np.mean((true_perclos[ve] - ea_pred[ve])**2)) if sum(ve)>0 else 0
    
    plt.plot(time_axis, base_pred, color='#d62728', linewidth=2.5, linestyle='--', label=f'Baseline Prediction (RMSE: {b_rmse:.3f})', alpha=0.8, zorder=2)
    plt.plot(time_axis, adabn_pred, color='#9467bd', linewidth=2.5, linestyle='-', label=f'AdaBN Prediction (RMSE: {a_rmse:.3f})', zorder=3)
    plt.plot(time_axis, ea_pred, color='#1f77b4', linewidth=2.5, linestyle='-.', label=f'EA Prediction (RMSE: {e_rmse:.3f})', zorder=4)
    
    plt.title("Model Prediction Agreement with True Fatigue (Subject 2)", weight='bold')
    plt.xlabel("Epoch Time (Chronological Order)", weight='bold')
    plt.ylabel("Fatigue / Probability", weight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved prediction overlay plot to {out_path}")

def plot_all_subject_overlays(baseline_csv_path: str, adabn_csv_path: str, ea_csv_path: str, out_dir: str):
    """Plot Baseline vs AdaBN vs EA predictions for every subject individually."""
    window_size = 50
    
    baseline_df = load_df(baseline_csv_path)
    adabn_df = load_df(adabn_csv_path)
    ea_df = load_df(ea_csv_path)
    
    subjects = baseline_df['subject_id'].unique()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    for subject_id in subjects:
        b_df = baseline_df[baseline_df['subject_id'] == subject_id].sort_values('epoch_index')
        a_df = adabn_df[adabn_df['subject_id'] == subject_id].sort_values('epoch_index')
        e_df = ea_df[ea_df['subject_id'] == subject_id].sort_values('epoch_index')
        
        if len(b_df) == 0:
            continue
            
        merged = b_df[['epoch_index', 'perclos', 'y_score']].rename(columns={'y_score': 'base_pred'})
        merged = merged.merge(a_df[['epoch_index', 'y_score']].rename(columns={'y_score': 'adabn_pred'}), on='epoch_index', how='left')
        merged = merged.merge(e_df[['epoch_index', 'y_score']].rename(columns={'y_score': 'ea_pred'}), on='epoch_index', how='left')
        
        time_axis = merged['epoch_index'].values
        true_perclos = smooth(merged['perclos'].values, window_size)
        base_pred = smooth(merged['base_pred'].values, window_size)
        adabn_pred = smooth(merged['adabn_pred'].values, window_size)
        ea_pred = smooth(merged['ea_pred'].values, window_size)
        
        plt.figure(figsize=(12, 6))
        plt.rcParams.update({'font.size': 14})
        
        plt.plot(time_axis, true_perclos, color='#2ca02c', linewidth=4, label='True PERCLOS (Fatigue)', zorder=1)
        
        v = ~np.isnan(true_perclos) & ~np.isnan(base_pred)
        b_rmse = np.sqrt(np.mean((true_perclos[v] - base_pred[v])**2)) if sum(v)>0 else 0
        va = ~np.isnan(true_perclos) & ~np.isnan(adabn_pred)
        a_rmse = np.sqrt(np.mean((true_perclos[va] - adabn_pred[va])**2)) if sum(va)>0 else 0
        ve = ~np.isnan(true_perclos) & ~np.isnan(ea_pred)
        e_rmse = np.sqrt(np.mean((true_perclos[ve] - ea_pred[ve])**2)) if sum(ve)>0 else 0
        
        plt.plot(time_axis, base_pred, color='#d62728', linewidth=2.5, linestyle='--', label=f'Baseline Prediction (RMSE: {b_rmse:.3f})', alpha=0.8, zorder=2)
        plt.plot(time_axis, adabn_pred, color='#9467bd', linewidth=2.5, linestyle='-', label=f'AdaBN Prediction (RMSE: {a_rmse:.3f})', zorder=3)
        plt.plot(time_axis, ea_pred, color='#1f77b4', linewidth=2.5, linestyle='-.', label=f'EA Prediction (RMSE: {e_rmse:.3f})', zorder=4)
        
        plt.title(f"Model Prediction Agreement with True Fatigue (Subject {subject_id})", weight='bold')
        plt.xlabel("Epoch Time (Chronological Order)", weight='bold')
        plt.ylabel("Fatigue / Probability", weight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        out_path = Path(out_dir) / f"subject_{subject_id}_overlay.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
    
    print(f"Saved {len(subjects)} individual subject overlay plots to {out_dir}")

def plot_average_prediction_overlay(baseline_csv_path: str, adabn_csv_path: str, ea_csv_path: str, out_path: str):
    """Plot Baseline vs AdaBN vs EA predictions overlaying true PERCLOS averaged across all subjects."""
    window_size = 50
    
    baseline_df = load_df(baseline_csv_path)
    adabn_df = load_df(adabn_csv_path)
    ea_df = load_df(ea_csv_path)
    
    # Average across all subjects grouped by epoch_index
    # Note: baseline_df has perclos and y_score. adabn_df has y_score.
    base_avg = baseline_df.groupby('epoch_index')[['perclos', 'y_score']].mean()
    adabn_avg = adabn_df.groupby('epoch_index')['y_score'].mean()
    ea_avg = ea_df.groupby('epoch_index')['y_score'].mean()
    
    merged = base_avg.rename(columns={'y_score': 'base_pred'})
    merged = merged.merge(adabn_avg.rename('adabn_pred'), left_index=True, right_index=True, how='left')
    merged = merged.merge(ea_avg.rename('ea_pred'), left_index=True, right_index=True, how='left')
    
    time_axis = merged.index.values
    true_perclos = smooth(merged['perclos'].values, window_size)
    base_pred = smooth(merged['base_pred'].values, window_size)
    adabn_pred = smooth(merged['adabn_pred'].values, window_size)
    ea_pred = smooth(merged['ea_pred'].values, window_size)
    
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(time_axis, true_perclos, color='#2ca02c', linewidth=4, label='True PERCLOS (Fatigue)', zorder=1)
    
    v = ~np.isnan(true_perclos) & ~np.isnan(base_pred)
    b_rmse = np.sqrt(np.mean((true_perclos[v] - base_pred[v])**2)) if sum(v)>0 else 0
    va = ~np.isnan(true_perclos) & ~np.isnan(adabn_pred)
    a_rmse = np.sqrt(np.mean((true_perclos[va] - adabn_pred[va])**2)) if sum(va)>0 else 0
    ve = ~np.isnan(true_perclos) & ~np.isnan(ea_pred)
    e_rmse = np.sqrt(np.mean((true_perclos[ve] - ea_pred[ve])**2)) if sum(ve)>0 else 0
    
    plt.plot(time_axis, base_pred, color='#d62728', linewidth=2.5, linestyle='--', label=f'Baseline Prediction (RMSE: {b_rmse:.3f})', alpha=0.8, zorder=2)
    plt.plot(time_axis, adabn_pred, color='#9467bd', linewidth=2.5, linestyle='-', label=f'AdaBN Prediction (RMSE: {a_rmse:.3f})', zorder=3)
    plt.plot(time_axis, ea_pred, color='#1f77b4', linewidth=2.5, linestyle='-.', label=f'EA Prediction (RMSE: {e_rmse:.3f})', zorder=4)
    
    plt.title("Average Model Prediction Agreement with True Fatigue (All Subjects)", weight='bold')
    plt.xlabel("Epoch Time (Chronological Order)", weight='bold')
    plt.ylabel("Fatigue / Probability", weight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved average prediction overlay plot to {out_path}")

def plot_global_prediction_distribution(baseline_csv_path: str, adabn_csv_path: str, ea_csv_path: str, out_path: str):
    baseline_df = load_df(baseline_csv_path)
    adabn_df = load_df(adabn_csv_path)
    ea_df = load_df(ea_csv_path)
    
    # Create bins for True PERCLOS
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2\n(Alert)', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0\n(Drowsy)']
    
    baseline_df['perclos_bin'] = pd.cut(baseline_df['perclos'], bins=bins, labels=labels, include_lowest=True)
    adabn_df['perclos_bin'] = pd.cut(baseline_df['perclos'], bins=bins, labels=labels, include_lowest=True) # use baseline's perclos to match
    ea_df['perclos_bin'] = pd.cut(baseline_df['perclos'], bins=bins, labels=labels, include_lowest=True)

    baseline_df['Method'] = 'Baseline'
    adabn_df['Method'] = 'AdaBN'
    ea_df['Method'] = 'EA'
    
    # Combine dfs for seaborn
    combined_df = pd.concat([
        baseline_df[['perclos_bin', 'y_score', 'Method']],
        adabn_df[['perclos_bin', 'y_score', 'Method']],
        ea_df[['perclos_bin', 'y_score', 'Method']]
    ])
    combined_df = combined_df.dropna(subset=['perclos_bin'])
    
    plt.figure(figsize=(14, 7))
    plt.rcParams.update({'font.size': 14})
    
    palette = {'Baseline': '#d62728', 'AdaBN': '#9467bd', 'EA': '#1f77b4'}
    sns.violinplot(data=combined_df, x='perclos_bin', y='y_score', hue='Method', 
                   palette=palette, inner='quartile', cut=0, linewidth=1.5, density_norm='width')
    
    plt.title("Global Prediction Distribution across True Fatigue States (All Epochs)", weight='bold')
    plt.xlabel("True PERCLOS Range", weight='bold')
    plt.ylabel("Predicted Probability of Drowsiness", weight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title='Adaptation Method', loc='upper left')
    plt.tight_layout()
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved global prediction distribution plot to {out_path}")

def plot_per_subject_adaptation_gain(out_path: str):
    """Plot per-subject adaptation gain relative to baseline."""
    # Load fold metrics
    base_df = pd.read_csv("data/results/experiments/Baselines/Deep/Cross-subject/EEGNET_Baseline/fold_metrics.csv")
    adabn_df = pd.read_csv("data/results/experiments/Adaptation/ADABN/adabn_f1.0/fold_metrics.csv")
    ea_df = pd.read_csv("data/results/experiments/Adaptation/EA/ea_f1.0/fold_metrics.csv")
    ea_adabn_df = pd.read_csv("data/results/experiments/Adaptation/BOTH_EA_ADABN/ea_adabn_f1.0/fold_metrics.csv")
    
    # Merge on test_subject_id
    base_df = base_df[['test_subject_id', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'base_acc'}).set_index('test_subject_id')
    adabn_df = adabn_df[['test_subject_id', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'adabn_acc'}).set_index('test_subject_id')
    ea_df = ea_df[['test_subject_id', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'ea_acc'}).set_index('test_subject_id')
    ea_adabn_df = ea_adabn_df[['test_subject_id', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'ea_adabn_acc'}).set_index('test_subject_id')
    
    df = base_df.join([adabn_df, ea_df, ea_adabn_df], how='inner').sort_index()
    
    # Calculate gains
    df['adabn_gain'] = df['adabn_acc'] - df['base_acc']
    df['ea_gain'] = df['ea_acc'] - df['base_acc']
    df['ea_adabn_gain'] = df['ea_adabn_acc'] - df['base_acc']
    
    subjects = [str(x) for x in df.index]
    x = np.arange(len(subjects))
    width = 0.25
    
    plt.figure(figsize=(16, 7))
    plt.rcParams.update({'font.size': 14})
    
    plt.bar(x - width, df['adabn_gain'], width, label='AdaBN', color='#9467bd', edgecolor='black', zorder=3)
    plt.bar(x, df['ea_gain'], width, label='EA', color='#1f77b4', edgecolor='black', zorder=3)
    plt.bar(x + width, df['ea_adabn_gain'], width, label='EA + AdaBN', color='#e377c2', edgecolor='black', zorder=3)
    
    plt.axhline(0, color='black', linewidth=2, zorder=4)
    plt.xlabel('Test Subject ID', weight='bold')
    plt.ylabel('Adaptation Gain ($\Delta$ Balanced Accuracy)', weight='bold')
    plt.title('Per-Subject Adaptation Gain vs. Baseline EEGNet', weight='bold', fontsize=16)
    plt.xticks(x, subjects)
    
    # Add a horizontal dashed line at Y=0 labeled Baseline
    plt.text(-1, 0.01, 'Baseline Performance (0)', va='bottom', ha='left', weight='bold', color='#7f7f7f', fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved per-subject adaptation gain plot to {out_path}")

if __name__ == '__main__':
    base_csv = "data/results/experiments/baseline_eegnet/predictions.csv"
    adabn_csv = "data/results/experiments/adabn_f1.0/predictions.csv"
    ea_csv = "data/results/experiments/ea_f1.0/predictions.csv"
    plot_prediction_overlay(base_csv, adabn_csv, ea_csv, "data/results/presentation/prediction_overlay.png")
    plot_average_prediction_overlay(base_csv, adabn_csv, ea_csv, "data/results/presentation/average_prediction_overlay.png")
    plot_all_subject_overlays(base_csv, adabn_csv, ea_csv, "data/results/presentation/subject_overlays")
    plot_global_prediction_distribution(base_csv, adabn_csv, ea_csv, "data/results/presentation/global_prediction_distribution.png")
    plot_per_subject_adaptation_gain("data/results/presentation/per_subject_adaptation_gain.png")
