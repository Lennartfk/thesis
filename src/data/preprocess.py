from src.data.seedvig_loader import RAW_DATASET_DIR, SeedVigLoader


def preprocess_recording(file_name, rootpath=None, l_freq=1.0, h_freq=50.0):
    loader = SeedVigLoader(rootpath=rootpath or RAW_DATASET_DIR, l_freq=l_freq, h_freq=h_freq)
    return loader.process_single_file(file_name)
