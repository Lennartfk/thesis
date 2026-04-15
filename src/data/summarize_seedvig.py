import argparse
from pathlib import Path

import mne
import numpy as np

from load_labels import load_perclos_labels
from seedvig_loader import EPOCHED_DIR, PROCESSED_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize processed SEED-VIG recordings and optionally save an MNE HTML report."
    )
    parser.add_argument(
        "--recording",
        help="Recording name without suffix, for example '1_20151124_noon_2'. Defaults to the first processed file.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save an HTML MNE report alongside the console summary.",
    )
    return parser.parse_args()


def resolve_recording_name(recording_name=None):
    if recording_name:
        return recording_name

    raw_files = sorted(PROCESSED_DIR.glob("*_raw.fif"))
    if not raw_files:
        raise FileNotFoundError(f"No processed raw FIF files found in {PROCESSED_DIR}")
    return raw_files[0].stem.replace("_raw", "")


def get_paths(recording_name):
    raw_path = PROCESSED_DIR / f"{recording_name}_raw.fif"
    epoch_path = EPOCHED_DIR / f"{recording_name}-epo.fif"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing processed raw file: {raw_path}")
    return raw_path, epoch_path


def print_raw_summary(raw):
    duration_seconds = raw.n_times / raw.info["sfreq"]
    print("\nRaw summary")
    print(raw)
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {duration_seconds:.2f} seconds")


def print_epoch_summary(epochs):
    print("\nEpoch summary")
    print(epochs)
    print(f"Epochs: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Sampling rate: {epochs.info['sfreq']} Hz")
    print(f"Epoch duration: {(epochs.tmax - epochs.tmin) + (1.0 / epochs.info['sfreq']):.2f} seconds")


def print_label_summary(labels):
    print("\nLabel summary")
    print(f"Labels: {labels.size}")
    print(f"Min PERCLOS: {labels.min():.4f}")
    print(f"Max PERCLOS: {labels.max():.4f}")
    print(f"Mean PERCLOS: {labels.mean():.4f}")
    print(f"Std PERCLOS: {labels.std():.4f}")


def build_report(recording_name, raw, epochs, labels, report_path):
    report = mne.Report(title=f"SEED-VIG Summary: {recording_name}")
    report.add_raw(raw=raw, title="Processed Raw EEG")

    fig_psd = raw.compute_psd().plot(show=False)
    report.add_figure(fig=fig_psd, title="Raw Power Spectral Density")

    fig_sensors = raw.plot_sensors(show=False)
    report.add_figure(fig=fig_sensors, title="Sensor Positions")

    if epochs is not None:
        report.add_epochs(epochs=epochs, title="Epoched EEG")
        fig_epoch_psd = epochs.compute_psd().plot(show=False)
        report.add_figure(fig=fig_epoch_psd, title="Epoch Power Spectral Density")

    label_html = f"""
    <h2>Label Summary</h2>
    <ul>
        <li><b>Recording:</b> {recording_name}</li>
        <li><b>Number of labels:</b> {labels.size}</li>
        <li><b>Minimum PERCLOS:</b> {labels.min():.4f}</li>
        <li><b>Maximum PERCLOS:</b> {labels.max():.4f}</li>
        <li><b>Mean PERCLOS:</b> {labels.mean():.4f}</li>
        <li><b>Std PERCLOS:</b> {labels.std():.4f}</li>
    </ul>
    """
    report.add_html(title="PERCLOS Labels", html=label_html)
    report.save(report_path, overwrite=True, open_browser=False)


def main():
    args = parse_args()
    recording_name = resolve_recording_name(args.recording)
    raw_path, epoch_path = get_paths(recording_name)

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
    labels = load_perclos_labels(recording_name)

    print(f"Recording: {recording_name}")
    print_raw_summary(raw)
    print_label_summary(labels)

    epochs = None
    if epoch_path.exists():
        epochs = mne.read_epochs(epoch_path, preload=False, verbose=False)
        print_epoch_summary(epochs)
        if len(epochs) != labels.size:
            print(
                f"\n[warn] Epoch/label count mismatch: {len(epochs)} epochs vs {labels.size} labels"
            )
    else:
        print(f"\nNo epoched file found at {epoch_path}")

    if args.save_report:
        reports_dir = Path(__file__).resolve().parents[2] / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{recording_name}_summary.html"
        build_report(recording_name, raw, epochs, labels, report_path)
        print(f"\nSaved MNE report to {report_path}")


if __name__ == "__main__":
    main()
