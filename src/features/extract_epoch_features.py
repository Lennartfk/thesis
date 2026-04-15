import argparse
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "SEED_VIG"
EPOCHED_DIR = DATA_DIR / "epoched"
FEATURES_DIR = DATA_DIR / "features"
DEFAULT_OUTPUT_PATH = FEATURES_DIR / "seedvig_spectral_features.csv"
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a minimal Yeo-inspired SEED-VIG feature table from epoched FIF files. "
            "Labels are read only from epoch metadata."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(EPOCHED_DIR),
        help="Directory containing *-epo.fif files with recording, epoch_index, and perclos metadata.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-time-features",
        action="store_true",
        help="Only write band-power features; omit per-channel mean and std.",
    )
    return parser.parse_args()


def parse_subject_id(recording_name):
    return recording_name.split("_", 1)[0]


def clean_column_token(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def require_metadata(epochs, epoch_path):
    required_columns = {"recording", "epoch_index", "perclos"}
    if epochs.metadata is None:
        raise ValueError(f"{epoch_path.name} has no metadata; rerun epoch_seedvig.py first.")

    missing_columns = required_columns.difference(epochs.metadata.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{epoch_path.name} metadata is missing required columns: {missing_text}")

    return epochs.metadata.reset_index(drop=True).copy()


def bandpower_features(epoch_psd, freqs, channel_names):
    features = {}
    for band_name, (fmin, fmax) in BANDS.items():
        band_mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(band_mask):
            raise ValueError(f"No PSD frequency bins found for {band_name} ({fmin}-{fmax} Hz).")

        band_freqs = freqs[band_mask]
        band_psd = epoch_psd[:, band_mask]
        powers = np.trapezoid(band_psd, band_freqs, axis=1)

        for channel_name, power in zip(channel_names, powers):
            features[f"{channel_name}_{band_name}_power"] = float(power)

    return features


def time_features(epoch_data, channel_names):
    features = {}
    for channel_name, signal in zip(channel_names, epoch_data):
        features[f"{channel_name}_mean"] = float(np.mean(signal))
        features[f"{channel_name}_std"] = float(np.std(signal))
    return features


def build_rows_for_file(epoch_path, include_time_features=True):
    epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
    epochs = epochs.copy().pick("eeg")
    metadata = require_metadata(epochs, epoch_path)

    recordings = metadata["recording"].astype(str)
    epoch_indices = metadata["epoch_index"].to_numpy(dtype=int)
    channel_names = [clean_column_token(name) for name in epochs.ch_names]
    epoch_data = epochs.get_data(copy=False)
    spectrum = epochs.compute_psd(fmin=1.0, fmax=30.0, verbose=False)
    psd_data = spectrum.get_data()
    freqs = spectrum.freqs

    rows = []
    for epoch_idx in range(len(epochs)):
        row = {
            "subject_id": parse_subject_id(str(recordings.iloc[epoch_idx])),
            "recording": str(recordings.iloc[epoch_idx]),
            "epoch_index": int(epoch_indices[epoch_idx]),
            "perclos": float(metadata.loc[epoch_idx, "perclos"]),
        }
        row.update(bandpower_features(psd_data[epoch_idx], freqs, channel_names))
        if include_time_features:
            row.update(time_features(epoch_data[epoch_idx], channel_names))
        rows.append(row)

    return rows


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    epoch_files = sorted(input_dir.glob("*-epo.fif"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoched FIF files found in {input_dir}")

    all_rows = []
    for epoch_path in epoch_files:
        print(f"Extracting spectral features from {epoch_path.name}...")
        all_rows.extend(
            build_rows_for_file(
                epoch_path,
                include_time_features=not args.no_time_features,
            )
        )

    feature_frame = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output_path, index=False)
    print(f"Saved {len(feature_frame)} epoch rows to {output_path}")


if __name__ == "__main__":
    main()
