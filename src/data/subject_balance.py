import numpy as np


SUBJECT_BALANCE_COLUMNS = [
    "subject_id",
    "n_alert",
    "n_drowsy",
    "n_total",
    "minority_count",
    "minority_fraction",
    "retained",
    "exclusion_reason",
]


def subject_class_balance(df, subject_column="subject_id", target_column="target"):
    balance = (
        df.groupby(subject_column)[target_column]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "n_alert", 1: "n_drowsy"})
        .reset_index()
    )

    for column in ["n_alert", "n_drowsy"]:
        if column not in balance.columns:
            balance[column] = 0

    balance = balance.rename(columns={subject_column: "subject_id"})
    balance["n_alert"] = balance["n_alert"].astype(int)
    balance["n_drowsy"] = balance["n_drowsy"].astype(int)
    balance["n_total"] = balance["n_alert"] + balance["n_drowsy"]
    balance["minority_count"] = balance[["n_alert", "n_drowsy"]].min(axis=1)
    balance["minority_fraction"] = np.where(
        balance["n_total"] > 0,
        balance["minority_count"] / balance["n_total"],
        np.nan,
    )
    return balance[["subject_id", "n_alert", "n_drowsy", "n_total", "minority_count", "minority_fraction"]]


def _exclusion_reason(row, min_subject_class_samples):
    missing = []
    if row["n_alert"] < min_subject_class_samples:
        missing.append("alert")
    if row["n_drowsy"] < min_subject_class_samples:
        missing.append("drowsy")
    if not missing:
        return ""
    return f"class_count_below_{min_subject_class_samples}:missing_{'_and_'.join(missing)}"


def filter_subjects_by_class_balance(
    df,
    min_subject_class_samples=1,
    subject_column="subject_id",
    target_column="target",
):
    if min_subject_class_samples < 0:
        raise ValueError("min_subject_class_samples must be non-negative.")

    balance = subject_class_balance(df, subject_column=subject_column, target_column=target_column)
    balance["retained"] = balance["minority_count"] >= int(min_subject_class_samples)
    balance["exclusion_reason"] = balance.apply(
        _exclusion_reason,
        axis=1,
        min_subject_class_samples=int(min_subject_class_samples),
    )
    balance = balance[SUBJECT_BALANCE_COLUMNS]

    retained_subjects = set(balance.loc[balance["retained"], "subject_id"].astype(str))
    filtered = df.loc[df[subject_column].astype(str).isin(retained_subjects)].copy()
    excluded = balance.loc[~balance["retained"]].copy()

    if filtered.empty:
        raise ValueError(
            "No subjects remain after class-balance filtering. "
            "Lower min_subject_class_samples or disable exclude_imbalanced_subjects."
        )

    return filtered, balance, excluded
