import re
from collections.abc import Iterable

import pandas as pd

TARGET = "default_next_month"
EXPECTED_ROW_COUNT = 30_000

ALLOWED_MODEL_FEATURES = (
    "LIMIT_BAL",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)
EXCLUDED_MODEL_FEATURES = ("SEX", "EDUCATION", "MARRIAGE", "AGE")
ALL_FEATURES = (
    "LIMIT_BAL",
    *EXCLUDED_MODEL_FEATURES,
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)

_X_ALIASES = dict(
    zip(
        (f"X{index}" for index in range(1, 24)),
        ALL_FEATURES,
        strict=True,
    )
)
_TARGET_ALIASES = {"Y", "DEFAULT_PAYMENT_NEXT_MONTH", "DEFAULT_NEXT_MONTH", TARGET.upper()}


def normalize_name(name: object) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")
    if normalized == "PAY_1":
        return "PAY_0"
    if normalized in _TARGET_ALIASES:
        return TARGET
    return _X_ALIASES.get(normalized, normalized)


def normalize_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame.columns = [normalize_name(column) for column in frame.columns]
    frame = frame.drop(columns=[column for column in ("ID", "INDEX") if column in frame])
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns after normalization: {duplicates}")

    missing = sorted(set(ALL_FEATURES) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(ALL_FEATURES))
    if missing or unexpected:
        raise ValueError(f"Dataset schema mismatch; missing={missing}, unexpected={unexpected}")

    frame = frame.loc[:, ALL_FEATURES]
    for column in ALL_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def normalize_target(target: pd.Series | pd.DataFrame | Iterable[int]) -> pd.Series:
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(f"Expected one target column, found {target.shape[1]}")
        series = target.iloc[:, 0]
    elif isinstance(target, pd.Series):
        series = target
    else:
        series = pd.Series(target)

    series = pd.to_numeric(series, errors="raise")
    if series.isna().any():
        raise ValueError("Target contains missing values")
    if not set(series.unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found {sorted(series.unique().tolist())}")
    return series.astype("int8").rename(TARGET).reset_index(drop=True)


def validate_processed_dataset(frame: pd.DataFrame, *, allow_subset: bool = False) -> None:
    expected_columns = [*ALL_FEATURES, TARGET]
    if frame.columns.tolist() != expected_columns:
        raise ValueError("Processed dataset columns or order do not match the canonical schema")
    if not allow_subset and len(frame) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(frame)}")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Processed target must contain only non-missing 0/1 values")
