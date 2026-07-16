from dataclasses import dataclass

from src.data.epochs import build_epoch_index
from src.data.features import load_labeled_features
from src.data.subject_balance import (
    SUBJECT_BALANCE_COLUMNS,
    filter_subjects_by_class_balance,
    subject_class_balance,
)


@dataclass
class PreparedDataset:
    df: object
    raw_df: object
    subject_balance: object
    excluded_subjects: object
    feature_columns: list
    dropped_columns: list
    feature_count: int | None = None


def apply_subject_balance_filter(df, config):
    if config.exclude_imbalanced_subjects:
        filtered_df, subject_balance, excluded_subjects = filter_subjects_by_class_balance(
            df,
            min_subject_class_samples=config.min_subject_class_samples,
        )
    else:
        subject_balance = subject_class_balance(df)
        subject_balance["retained"] = True
        subject_balance["exclusion_reason"] = ""
        subject_balance = subject_balance[SUBJECT_BALANCE_COLUMNS]
        filtered_df = df.copy()
        excluded_subjects = subject_balance.loc[~subject_balance["retained"]].copy()

    if not excluded_subjects.empty:
        excluded_ids = ",".join(excluded_subjects["subject_id"].astype(str).tolist())
        print(
            f"Excluded {len(excluded_subjects)} subject(s) below "
            f"{config.min_subject_class_samples} sample(s) per class: {excluded_ids}"
        )
    return filtered_df, subject_balance, excluded_subjects


def prepare_feature_dataset(config):
    raw_df, feature_columns, dropped_columns = load_labeled_features(
        config.feature_path,
        alert_threshold=config.alert_threshold,
        drowsy_threshold=config.drowsy_threshold,
    )
    df, subject_balance, excluded_subjects = apply_subject_balance_filter(raw_df, config)

    return PreparedDataset(
        df=df,
        raw_df=raw_df,
        subject_balance=subject_balance,
        excluded_subjects=excluded_subjects,
        feature_columns=feature_columns,
        dropped_columns=dropped_columns,
        feature_count=len(feature_columns),
    )


def prepare_epoch_dataset(config):
    raw_df = build_epoch_index(
        config.epoch_dir,
        alert_threshold=config.alert_threshold,
        drowsy_threshold=config.drowsy_threshold,
    )
    df, subject_balance, excluded_subjects = apply_subject_balance_filter(raw_df, config)

    return PreparedDataset(
        df=df,
        raw_df=raw_df,
        subject_balance=subject_balance,
        excluded_subjects=excluded_subjects,
        feature_columns=[],
        dropped_columns=[],
    )
