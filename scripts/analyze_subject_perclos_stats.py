import pandas as pd
import matplotlib.pyplot as plt
import mne
import numpy as np
from pathlib import Path
import seaborn as sns

def parse_subject_id(recording_name):
    rec_str = str(recording_name)
    base_id = rec_str.split("_", 1)[0]
    
    if "4_20151105" in rec_str: return "4_1"
    if "4_20151107" in rec_str: return "4_2"
    if "5_20141108" in rec_str: return "5_1"
    if "5_20151012" in rec_str: return "5_2"
    
    return base_id

def analyze_stats(out_dir="data/results/Dataset_Stats", include_title=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = Path("data/SEED_VIG/epoched")
    epoch_paths = sorted(epoch_dir.glob("*-epo.fif"))
    
    alert_threshold = 0.35
    drowsy_threshold = 0.70
    
    results = []
    all_metadata = []

    print("Extracting full PERCLOS statistics (including intermediate epochs)...")
    for epoch_path in epoch_paths:
        try:
            epochs = mne.read_epochs(epoch_path, preload=False, verbose=False)
            metadata = epochs.metadata.copy()
            
            metadata['subject_id'] = metadata['recording'].apply(parse_subject_id)
            all_metadata.append(metadata)
            print(f"Loaded {epoch_path.name}")
        except Exception as e:
            print(f"Failed to load {epoch_path.name}: {e}")

    full_df = pd.concat(all_metadata, ignore_index=True)
    
    subjects = full_df['subject_id'].unique()
    
    for sub in subjects:
        sub_perclos = full_df[full_df['subject_id'] == sub]['perclos']
        
        n_epochs = len(sub_perclos)
        time_minutes = (n_epochs * 8) / 60.0
        
        alert_mask = sub_perclos <= alert_threshold
        drowsy_mask = sub_perclos >= drowsy_threshold
        inter_mask = (~alert_mask) & (~drowsy_mask)
        
        alert_frac = alert_mask.mean()
        drowsy_frac = drowsy_mask.mean()
        inter_frac = inter_mask.mean()
        
        results.append({
            'Subject_ID': sub,
            'Total_Epochs': n_epochs,
            'Total_Time_Minutes': round(time_minutes, 1),
            'Mean_PERCLOS': round(sub_perclos.mean(), 3),
            'Std_PERCLOS': round(sub_perclos.std(), 3),
            'Min_PERCLOS': round(sub_perclos.min(), 3),
            'Max_PERCLOS': round(sub_perclos.max(), 3),
            'Percent_Alert': round(alert_frac * 100, 1),
            'Percent_Intermediate': round(inter_frac * 100, 1),
            'Percent_Drowsy': round(drowsy_frac * 100, 1)
        })

    results_df = pd.DataFrame(results)
    
    def sort_key(s):
        parts = str(s).split('_')
        return float(parts[0]) + (float(parts[1])/10 if len(parts) > 1 else 0)
        
    results_df['Subject_ID_Num'] = results_df['Subject_ID'].apply(sort_key)
    results_df = results_df.sort_values('Subject_ID_Num').drop(columns=['Subject_ID_Num'])
    
    full_df['Subject_ID_Num'] = full_df['subject_id'].apply(sort_key)
    full_df = full_df.sort_values('Subject_ID_Num')
    
    out_path_csv = out_dir / "subject_perclos_statistics_full.csv"
    results_df.to_csv(out_path_csv, index=False)
    print(f"\nSaved statistics to {out_path_csv}")
    
    plt.figure(figsize=(12, 6))
    subjects_sorted = results_df['Subject_ID'].astype(str)
    p_alert = results_df['Percent_Alert']
    p_inter = results_df['Percent_Intermediate']
    p_drowsy = results_df['Percent_Drowsy']
    
    plt.bar(subjects_sorted, p_alert, color='#2ca02c', label='Alert (PERCLOS ≤ 0.35)')
    plt.bar(subjects_sorted, p_inter, bottom=p_alert, color='#ff7f0e', label='Intermediate (0.35 < P < 0.70)')
    plt.bar(subjects_sorted, p_drowsy, bottom=p_alert + p_inter, color='#d62728', label='Drowsy (PERCLOS ≥ 0.70)')
    
    if include_title:
        plt.title('Distribution of All PERCLOS States per Subject (Unfiltered Data)', weight='bold', fontsize=14)
    plt.xlabel('Subject ID', weight='bold', fontsize=12)
    plt.ylabel('Percentage of Total Time (%)', weight='bold', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_path_bar = out_dir / "subject_perclos_distribution_full.pdf"
    plt.savefig(out_path_bar, format="pdf", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='subject_id', y='perclos', data=full_df, color='lightblue', fliersize=2)
    plt.axhline(y=0.35, color='#2ca02c', linestyle='--', label='Alert Threshold')
    plt.axhline(y=0.70, color='#d62728', linestyle='--', label='Drowsy Threshold')
    
    if include_title:
        plt.title('Continuous PERCLOS Distribution per Subject (Unfiltered Data)', weight='bold', fontsize=14)
    plt.xlabel('Subject ID', weight='bold', fontsize=12)
    plt.ylabel('PERCLOS Value', weight='bold', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    out_path_box = out_dir / "subject_perclos_boxplot_full.pdf"
    plt.savefig(out_path_box, format="pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved figures to {out_dir}")
    
    plt.figure(figsize=(12, 6))
    
    filtered_total = p_alert + p_drowsy
    p_alert_norm = np.where(filtered_total > 0, (p_alert / filtered_total) * 100, 0)
    p_drowsy_norm = np.where(filtered_total > 0, (p_drowsy / filtered_total) * 100, 0)
    
    plt.bar(subjects_sorted, p_alert_norm, color='#2ca02c', label='Alert (PERCLOS ≤ 0.35)')
    plt.bar(subjects_sorted, p_drowsy_norm, bottom=p_alert_norm, color='#d62728', label='Drowsy (PERCLOS ≥ 0.70)')
    
    if include_title:
        plt.title('Distribution of PERCLOS States per Subject (Filtered Data)', weight='bold', fontsize=14)
    plt.xlabel('Subject ID', weight='bold', fontsize=12)
    plt.ylabel('Percentage of Filtered Time (%)', weight='bold', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_path_filtered = out_dir / "subject_perclos_distribution.pdf"
    plt.savefig(out_path_filtered, format="pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_stats()
