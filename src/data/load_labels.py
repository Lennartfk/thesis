from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from seedvig_loader import LABELS_DIR


def get_label_path(recording_name, labels_dir=LABELS_DIR):
    return Path(labels_dir) / f"{recording_name}.mat"


def load_perclos_labels(recording_name, labels_dir=LABELS_DIR):
    label_path = get_label_path(recording_name, labels_dir=labels_dir)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    data = loadmat(label_path)
    if "perclos" not in data:
        raise KeyError(f"'perclos' not found in {label_path}")

    return np.asarray(data["perclos"]).reshape(-1).astype(float)


def build_label_frame(recording_name, labels_dir=LABELS_DIR):
    labels = load_perclos_labels(recording_name, labels_dir=labels_dir)
    return pd.DataFrame(
        {
            "recording": recording_name,
            "epoch_index": np.arange(labels.size, dtype=int),
            "perclos": labels,
        }
    )


def load_all_perclos_labels(labels_dir=LABELS_DIR):
    labels_dir = Path(labels_dir)
    return {
        label_path.stem: load_perclos_labels(label_path.stem, labels_dir=labels_dir)
        for label_path in sorted(labels_dir.glob("*.mat"))
    }


def main():
    all_labels = load_all_perclos_labels()
    print(f"Loaded labels for {len(all_labels)} recordings from {LABELS_DIR}")

    if not all_labels:
        return

    first_recording = next(iter(all_labels))
    first_labels = all_labels[first_recording]
    print(
        f"Example recording: {first_recording}, "
        f"n_labels={first_labels.size}, "
        f"min={first_labels.min():.3f}, max={first_labels.max():.3f}"
    )


if __name__ == "__main__":
    main()
