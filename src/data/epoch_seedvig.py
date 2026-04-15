import argparse

import mne
import numpy as np
import pandas as pd

from load_labels import load_perclos_labels
from seedvig_loader import EPOCHED_DIR, PROCESSED_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create fixed-length 8-second epochs from preprocessed SEED-VIG raw FIF files."
    )
    parser.add_argument("--epoch-duration", type=float, default=8.0, help="Epoch length in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild epoch files that already exist.")
    return parser.parse_args()


def make_metadata(recording_name, n_epochs):
    labels = load_perclos_labels(recording_name)
    if labels.size != n_epochs:
        print(
            f"[warn] {recording_name}: found {labels.size} labels for {n_epochs} epochs, "
            f"trimming to {min(labels.size, n_epochs)}"
        )

    n_valid_epochs = min(labels.size, n_epochs)
    metadata = pd.DataFrame(
        {
            "recording": recording_name,
            "epoch_index": np.arange(n_valid_epochs, dtype=int),
            "perclos": labels[:n_valid_epochs],
        }
    )
    return metadata, n_valid_epochs


def epoch_single_file(raw_path, epoch_duration):
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    recording_name = raw_path.stem.replace("_raw", "")
    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_duration)

    metadata, n_valid_epochs = make_metadata(recording_name, len(events))
    events = events[:n_valid_epochs]

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"fixed_length": 1},
        tmin=0.0,
        tmax=epoch_duration - (1.0 / raw.info["sfreq"]),
        baseline=None,
        preload=True,
        metadata=metadata,
        verbose=False,
    )
    return epochs


def main():
    args = parse_args()
    EPOCHED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(PROCESSED_DIR.glob("*_raw.fif"))
    if not raw_files:
        raise FileNotFoundError(f"No processed FIF files found in {PROCESSED_DIR}")

    created_count = 0
    skipped_count = 0

    for raw_path in raw_files:
        save_path = EPOCHED_DIR / raw_path.name.replace("_raw.fif", "-epo.fif")
        if save_path.exists() and not args.overwrite:
            print(f"[skip] {save_path.name} already exists")
            skipped_count += 1
            continue

        print(f"Epoching {raw_path.name}...")
        epochs = epoch_single_file(raw_path, epoch_duration=args.epoch_duration)
        epochs.save(save_path, overwrite=True)
        print(f"[saved] {save_path.name}")
        created_count += 1

    print(f"Finished epoching. Created: {created_count}, skipped: {skipped_count}.")


if __name__ == "__main__":
    main()
