import os
import pandas as pd
from pathlib import Path
import numpy as np

def merge_ea_fraction_sweep():
    base = Path("data/results/experiments")
    ea_base = base / "Adaptation" / "EA"
    
    out_dir = base / "baseline_eegnet_ea_cv_sweep"
    os.makedirs(out_dir, exist_ok=True)
    
    all_fold_metrics = []
    
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        frac_dir = ea_base / f"ea_f{frac}"
        metric_file = frac_dir / "fold_metrics.csv"
        
        if metric_file.exists():
            df = pd.read_csv(metric_file)
            
            # The older format might not have adaptation_target_fraction.
            if "adaptation_target_fraction" not in df.columns:
                df["adaptation_target_fraction"] = frac
            if "adaptation_name" not in df.columns:
                df["adaptation_name"] = "ea"
                
            all_fold_metrics.append(df)
            
    if all_fold_metrics:
        merged_folds = pd.concat(all_fold_metrics, ignore_index=True)
        merged_folds.to_csv(out_dir / "fold_metrics.csv", index=False)
        print(f"Successfully merged EA fraction sweeps into {out_dir}")
        
        # Also generate summary.csv
        summary_rows = []
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            frac_df = merged_folds[np.isclose(merged_folds["adaptation_target_fraction"], frac)]
            if not frac_df.empty:
                # calculate mean across subjects/folds
                sum_row = frac_df.mean(numeric_only=True).to_dict()
                sum_row["adaptation_name"] = "ea"
                sum_row["adaptation_target_fraction"] = frac
                summary_rows.append(sum_row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "summary.csv", index=False)
    else:
        print("No EA fraction fold metrics found to merge.")

if __name__ == "__main__":
    merge_ea_fraction_sweep()
