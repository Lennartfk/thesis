import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    base_csv = "data/results/experiments/baseline_eegnet/predictions.csv"
    adabn_csv = "data/results/experiments/adabn_f1.0/predictions.csv"
    ea_csv = "data/results/experiments/ea_f1.0/predictions.csv"

    baseline_df = pd.read_csv(base_csv)
    adabn_df = pd.read_csv(adabn_csv)
    ea_df = pd.read_csv(ea_csv)

    # Average across subjects
    base_avg = baseline_df.groupby('epoch_index')[['perclos', 'y_score']].mean()
    adabn_avg = adabn_df.groupby('epoch_index')['y_score'].mean()
    ea_avg = ea_df.groupby('epoch_index')['y_score'].mean()

    # Align them exactly by index
    df = pd.DataFrame({
        'true_perclos': base_avg['perclos'],
        'base_pred': base_avg['y_score'],
        'adabn_pred': adabn_avg,
        'ea_pred': ea_avg
    }).dropna()

    true_p = df['true_perclos'].values
    
    print(f"{'Method':<12} | {'Pearson Correlation (r)':<23} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 65)

    for method, col in zip(["Baseline", "AdaBN", "EA"], ['base_pred', 'adabn_pred', 'ea_pred']):
        pred = df[col].values
        
        r, p_val = pearsonr(true_p, pred)
        rmse = np.sqrt(mean_squared_error(true_p, pred))
        mae = mean_absolute_error(true_p, pred)
        
        print(f"{method:<12} | {r:<23.4f} | {rmse:<10.4f} | {mae:<10.4f}")

if __name__ == '__main__':
    main()
