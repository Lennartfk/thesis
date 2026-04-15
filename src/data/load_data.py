import numpy as np
import pandas as pd


METADATA_COLUMNS = {"subject_id", "recording", "epoch_index", "perclos", "target"}


def load_feature_table(feature_path):
    return pd.read_csv(feature_path)


def add_binary_target(df, alert_threshold, drowsy_threshold):
    if "perclos" not in df.columns:
        raise ValueError("Feature table must contain perclos from epoch metadata.")

    alert_mask = df["perclos"] <= alert_threshold
    drowsy_mask = df["perclos"] >= drowsy_threshold
    filtered = df.loc[alert_mask | drowsy_mask].copy()
    filtered["target"] = np.where(filtered["perclos"] <= alert_threshold, 0, 1).astype(int)
    return filtered


def get_feature_columns(df):
    candidates = [col for col in df.columns if col not in METADATA_COLUMNS]
    numeric_columns = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    dropped = sorted(set(candidates).difference(numeric_columns))
    return numeric_columns, dropped


def load_labeled_features(feature_path, alert_threshold, drowsy_threshold):
    df = load_feature_table(feature_path)
    df = add_binary_target(df, alert_threshold, drowsy_threshold)
    feature_columns, dropped_columns = get_feature_columns(df)
    if df.empty:
        raise ValueError("No samples remain after thresholding PERCLOS into alert/drowsy classes.")
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")
    return df, feature_columns, dropped_columns
