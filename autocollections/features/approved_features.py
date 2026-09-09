import numpy as np
import pandas as pd

from autocollections.data.schema import ALLOWED_MODEL_FEATURES, EXCLUDED_MODEL_FEATURES

DERIVED_MODEL_FEATURES = (
    "latest_utilization",
    "mean_utilization_6m",
    "max_utilization_6m",
    "payment_to_bill_ratio_1m",
    "mean_payment_to_bill_ratio_6m",
    "repayment_delay_mean",
    "repayment_delay_max",
    "repayment_delay_recent",
    "repayment_delay_trend",
    "bill_trend",
    "payment_trend",
    "months_with_positive_delay",
    "months_with_payment",
)


def approved_feature_view(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ALLOWED_MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing approved model features: {missing}")

    approved = frame.loc[:, ALLOWED_MODEL_FEATURES].copy()
    leaked = sorted(set(approved.columns) & set(EXCLUDED_MODEL_FEATURES))
    if leaked:
        raise ValueError(f"Excluded demographic features in official view: {leaked}")
    return approved


def approved_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    raw = approved_feature_view(frame)
    limit = raw["LIMIT_BAL"].clip(lower=1)
    bills = raw[[f"BILL_AMT{month}" for month in range(1, 7)]]
    payments = raw[[f"PAY_AMT{month}" for month in range(1, 7)]]
    delays = raw[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].clip(lower=0)

    utilization = bills.clip(lower=0).div(limit, axis=0).clip(upper=5)
    payment_ratios = payments.div(bills.clip(lower=1).to_numpy()).clip(lower=0, upper=5)
    derived = pd.DataFrame(index=raw.index)
    derived["latest_utilization"] = utilization["BILL_AMT1"]
    derived["mean_utilization_6m"] = utilization.mean(axis=1)
    derived["max_utilization_6m"] = utilization.max(axis=1)
    derived["payment_to_bill_ratio_1m"] = payment_ratios["PAY_AMT1"]
    derived["mean_payment_to_bill_ratio_6m"] = payment_ratios.mean(axis=1)
    derived["repayment_delay_mean"] = delays.mean(axis=1)
    derived["repayment_delay_max"] = delays.max(axis=1)
    derived["repayment_delay_recent"] = delays["PAY_0"]
    derived["repayment_delay_trend"] = delays["PAY_0"] - delays["PAY_6"]
    derived["bill_trend"] = ((bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit).clip(-5, 5)
    derived["payment_trend"] = ((payments["PAY_AMT1"] - payments["PAY_AMT6"]) / limit).clip(-5, 5)
    derived["months_with_positive_delay"] = (delays > 0).sum(axis=1)
    derived["months_with_payment"] = (payments > 0).sum(axis=1)
    return derived.loc[:, DERIVED_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)


def approved_feature_matrix(frame: pd.DataFrame, *, include_derived: bool) -> pd.DataFrame:
    raw = approved_feature_view(frame)
    if not include_derived:
        return raw
    return pd.concat([raw, approved_derived_features(raw)], axis=1)
