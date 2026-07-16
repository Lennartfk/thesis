import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.cm as cm

def plot_3d(base_dir, method="ea", include_title=True, out_path=None):
    base_dir = Path(base_dir)
    
    method_dir = f"final_eval_{method}_chronological_sweep"
    pred_file = base_dir / method_dir / "predictions.csv"
    
    if not pred_file.exists():
        print(f"File not found: {pred_file}")
        return
        
    df = pd.read_csv(pred_file)
    df["epoch_minutes"] = df["epoch_index"] * 8 / 60.0
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(float)
    
    # X: Time since start of session (evaluation time)
    # Y: Calibration period (chronological_minutes)
    # Z: Accuracy
    
    bin_size = 10
    df["time_bin"] = (df["epoch_minutes"] // bin_size) * bin_size + (bin_size / 2)
    
    # Group by Calibration Period (Y) and Time Bin (X)
    agg_df = df.groupby(["chronological_minutes", "time_bin"])["correct"].agg(["mean", "count"]).reset_index()
    
    # Filter out bins with too few samples
    # And filter out bins that are BEFORE the calibration period (model isn't predicting those in this sweep)
    agg_df = agg_df[agg_df["time_bin"] > agg_df["chronological_minutes"]]
    agg_df = agg_df[agg_df["count"] >= 10]
    
    # Pivot to create a 2D grid for X and Y
    pivot_df = agg_df.pivot(index="chronological_minutes", columns="time_bin", values="mean")
    
    # It's possible we have NaNs in the grid if certain bins didn't have data.
    # We can interpolate or leave them. plot_surface handles NaNs by masking.
    
    X = pivot_df.columns.values
    Y = pivot_df.index.values
    X, Y = np.meshgrid(X, Y)
    Z = pivot_df.values
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', alpha=0.9,
                           linewidth=0.5, antialiased=True)
                           
    ax.set_xlabel("Evaluation Time since start (Minutes)", fontsize=12, labelpad=15)
    ax.set_ylabel("Calibration Period (Minutes)", fontsize=12, labelpad=15)
    ax.set_zlabel("Classification Accuracy", fontsize=12, labelpad=15)
    if include_title:
        title_method = "Euclidean Alignment" if method == "ea" else "AdaBN"
        ax.set_title(f"3D Accuracy Surface (EEGNet + {title_method})", fontsize=18)
    
    # Adjust viewing angle for best presentation
    ax.view_init(elev=30, azim=135)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Accuracy", pad=0.1)
    
    if out_path is None:
        out_path = "data/results/presentation_new/3d_accuracy_surface.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Successfully generated 3D plot at {out_path}")

if __name__ == "__main__":
    plot_3d("data/results/experiments/23_Subjects_Stable")
