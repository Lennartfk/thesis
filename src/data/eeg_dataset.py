from pathlib import Path

import mne
import numpy as np
import pandas as pd


def parse_subject_id(recording_name):
    return str(recording_name).split("_", 1)[0]


def build_epoch_index(epoch_dir, alert_threshold, drowsy_threshold):
    """Build a lightweight table pointing to retained epochs in FIF files."""
    epoch_paths = sorted(Path(epoch_dir).glob("*-epo.fif"))
    if not epoch_paths:
        raise FileNotFoundError(f"No epoched FIF files found in {epoch_dir}")

    rows = []
    for epoch_path in epoch_paths:
        epochs = mne.read_epochs(epoch_path, preload=False, verbose=False)
        if epochs.metadata is None:
            raise ValueError(f"{epoch_path.name} has no metadata.")

        metadata = epochs.metadata.reset_index(drop=True).copy()
        required = {"recording", "epoch_index", "perclos"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"{epoch_path.name} metadata missing columns: {sorted(missing)}")

        for row_in_file, metadata_row in metadata.iterrows():
            perclos = float(metadata_row["perclos"])
            if perclos <= alert_threshold:
                target = 0
            elif perclos >= drowsy_threshold:
                target = 1
            else:
                continue

            recording = str(metadata_row["recording"])
            rows.append(
                {
                    "epoch_path": str(epoch_path),
                    "row_in_file": int(row_in_file),
                    "subject_id": parse_subject_id(recording),
                    "recording": recording,
                    "epoch_index": int(metadata_row["epoch_index"]),
                    "perclos": perclos,
                    "target": target,
                }
            )

    index = pd.DataFrame(rows)
    if index.empty:
        raise ValueError("No retained epochs after PERCLOS thresholding.")
    return index


def infer_epoch_shape(epoch_dir):
    epoch_path = next(Path(epoch_dir).glob("*-epo.fif"))
    epochs = mne.read_epochs(epoch_path, preload=True, verbose=False).pick("eeg")
    return len(epochs.ch_names), len(epochs.times), float(epochs.info["sfreq"]), list(epochs.ch_names)


def load_epochs_from_index(index_df):
    """Materialize selected epochs as float32 tensor array with shape n x channels x samples."""
    arrays = []
    metadata_frames = []

    for epoch_path, group in index_df.groupby("epoch_path", sort=False):
        epochs = mne.read_epochs(epoch_path, preload=True, verbose=False).pick("eeg")
        positions = group["row_in_file"].to_numpy(dtype=int)
        data = epochs.get_data(copy=True)[positions].astype(np.float32, copy=False)
        arrays.append(data)
        metadata_frames.append(group.reset_index(drop=True))

    X = np.concatenate(arrays, axis=0)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    y = metadata["target"].to_numpy(dtype=np.int64)
    return X, y, metadata


class ChannelStandardizer:
    """Channel-wise normalization fit on training epochs only."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32, copy=False)
