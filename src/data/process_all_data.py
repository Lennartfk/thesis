import argparse

from seedvig_loader import PROCESSED_DIR, RAW_DATASET_DIR, SeedVigLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SEED-VIG raw MATLAB files into preprocessed MNE FIF files."
    )
    parser.add_argument("--overwrite", action="store_true", help="Reprocess files that already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    loader = SeedVigLoader(RAW_DATASET_DIR)
    file_names = loader.get_file_names()
    print(f"Found {len(file_names)} files to process.\n")

    processed_count = 0
    skipped_count = 0

    for file_name in file_names:
        save_path = PROCESSED_DIR / loader.get_processed_filename(file_name)
        if save_path.exists() and not args.overwrite:
            print(f"[skip] {file_name} -> {save_path.name} already exists")
            skipped_count += 1
            continue

        print(f"Processing {file_name}...")
        raw = loader.process_single_file(file_name)
        if raw is None:
            continue

        raw.save(save_path, overwrite=True)
        print(f"[saved] {save_path.name}\n")
        processed_count += 1

    print(
        "Finished preprocessing SEED-VIG raw recordings. "
        f"Processed: {processed_count}, skipped: {skipped_count}."
    )


if __name__ == "__main__":
    main()
