from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "SEED_VIG"
RAW_DATASET_DIR = DATA_DIR / "raw" / "SEED-VIG"
RAW_DATA_DIR = RAW_DATASET_DIR / "Raw_Data"
LABELS_DIR = RAW_DATASET_DIR / "perclos_labels"
PROCESSED_DIR = DATA_DIR / "processed"
EPOCHED_DIR = DATA_DIR / "epoched"
FEATURES_DIR = DATA_DIR / "features"


class SeedVigLoader:
    def __init__(self, rootpath=RAW_DATASET_DIR, l_freq=1.0, h_freq=50.0):
        self.rootpath = Path(rootpath)
        self.folder_path = self.rootpath / "Raw_Data"
        self.locs_file = self.rootpath / "channel_62_pos.locs"
        self.l_freq = l_freq
        self.h_freq = h_freq

        self.montage = mne.channels.read_custom_montage(self.locs_file)
        self.channel_renames = {"PZ": "Pz", "POZ": "POz", "OZ": "Oz"}

    @staticmethod
    def load_matlab_file(file_path):
        try:
            return loadmat(file_path)
        except Exception as exc:
            print(f"Error loading MATLAB file {file_path}: {exc}")
            return None

    @staticmethod
    def format_names(names):
        return [str(array[0]).strip() for array in names[0]]

    @staticmethod
    def get_processed_filename(file_name):
        return Path(file_name).stem + "_raw.fif"

    @staticmethod
    def get_recording_name(file_name):
        return Path(file_name).stem.replace("_raw", "")

    def simplify_matlab_data(self, data):
        eeg_data = data["EEG"]["data"][0, 0]
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        eeg_names = self.format_names(data["EEG"]["chn"][0, 0])
        eeg_sample_rate = float(data["EEG"]["sample_rate"][0, 0].item())

        return {
            "eeg": eeg_data.astype(np.float64) * 1e-6,
            "eeg_names": eeg_names,
            "eeg_sample_rate": eeg_sample_rate,
        }

    def get_file_names(self):
        return sorted(file_path.name for file_path in self.folder_path.glob("*.mat"))

    def load_single_file(self, file_name):
        file_path = self.folder_path / file_name
        data = self.load_matlab_file(file_path)
        if data is None:
            return None
        return self.simplify_matlab_data(data)

    def create_mne_raw(self, data):
        info = mne.create_info(
            ch_names=data["eeg_names"],
            sfreq=data["eeg_sample_rate"],
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data["eeg"], info, verbose=False)
        raw.rename_channels(self.channel_renames)
        raw.set_montage(self.montage)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)
        return raw

    def process_single_file(self, file_name):
        data = self.load_single_file(file_name)
        if data is None:
            return None

        raw = self.create_mne_raw(data)
        print(f"Processed {file_name}")
        return raw

    def load_folder(self):
        return [self.process_single_file(file_name) for file_name in self.get_file_names()]
