import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from autocollections.data.schema import ALLOWED_MODEL_FEATURES, EXCLUDED_MODEL_FEATURES
from autocollections.features.approved_features import approved_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/frozen_exp039_model.joblib"
MANIFEST_PATH = ROOT / "artifacts/frozen_exp039_model_manifest.json"
EXPECTED_MODEL_SHA256 = "dbe9e60bc85cdb432771ee7fa88119d6a89820b36213c16e031b5bdb41b5e0c4"
EXPECTED_TRAIN_SHA256 = "f11c6817911ba5068a46f6321493412a0af13531f5becd33828e18dfbe065ed3"
ACTION_NAMES = {
    0: "NO_CONTACT",
    1: "DIGITAL_REMINDER",
    2: "PAYMENT_PLAN_REVIEW",
    3: "HUMAN_ESCALATION",
}
SCENARIOS = {
    "Low-risk / strong payer": {
        "LIMIT_BAL": 300_000,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -1,
        "PAY_6": -1,
        "BILL_AMT1": 30_000,
        "BILL_AMT2": 32_000,
        "BILL_AMT3": 31_000,
        "BILL_AMT4": 29_000,
        "BILL_AMT5": 28_000,
        "BILL_AMT6": 27_000,
        "PAY_AMT1": 25_000,
        "PAY_AMT2": 24_000,
        "PAY_AMT3": 25_000,
        "PAY_AMT4": 23_000,
        "PAY_AMT5": 24_000,
        "PAY_AMT6": 22_000,
    },
    "Moderate-risk customer": {
        "LIMIT_BAL": 200_000,
        "PAY_0": 1,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": -1,
        "PAY_6": -1,
        "BILL_AMT1": 120_000,
        "BILL_AMT2": 116_000,
        "BILL_AMT3": 110_000,
        "BILL_AMT4": 103_000,
        "BILL_AMT5": 98_000,
        "BILL_AMT6": 94_000,
        "PAY_AMT1": 15_000,
        "PAY_AMT2": 14_000,
        "PAY_AMT3": 13_000,
        "PAY_AMT4": 12_000,
        "PAY_AMT5": 12_000,
        "PAY_AMT6": 11_000,
    },
    "High-utilization customer": {
        "LIMIT_BAL": 100_000,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 1,
        "PAY_4": 1,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 98_000,
        "BILL_AMT2": 96_000,
        "BILL_AMT3": 94_000,
        "BILL_AMT4": 91_000,
        "BILL_AMT5": 88_000,
        "BILL_AMT6": 85_000,
        "PAY_AMT1": 5_000,
        "PAY_AMT2": 5_000,
        "PAY_AMT3": 4_000,
        "PAY_AMT4": 4_000,
        "PAY_AMT5": 3_000,
        "PAY_AMT6": 3_000,
    },
    "High-risk / high-exposure": {
        "LIMIT_BAL": 500_000,
        "PAY_0": 3,
        "PAY_2": 3,
        "PAY_3": 2,
        "PAY_4": 2,
        "PAY_5": 2,
        "PAY_6": 1,
        "BILL_AMT1": 250_000,
        "BILL_AMT2": 245_000,
        "BILL_AMT3": 238_000,
        "BILL_AMT4": 230_000,
        "BILL_AMT5": 220_000,
        "BILL_AMT6": 210_000,
        "PAY_AMT1": 1_000,
        "PAY_AMT2": 2_000,
        "PAY_AMT3": 1_500,
        "PAY_AMT4": 1_000,
        "PAY_AMT5": 1_000,
        "PAY_AMT6": 1_000,
    },
    "Recovering customer": {
        "LIMIT_BAL": 180_000,
        "PAY_0": 0,
        "PAY_2": 1,
        "PAY_3": 2,
        "PAY_4": 2,
        "PAY_5": 3,
        "PAY_6": 3,
        "BILL_AMT1": 90_000,
        "BILL_AMT2": 105_000,
        "BILL_AMT3": 118_000,
        "BILL_AMT4": 132_000,
        "BILL_AMT5": 143_000,
        "BILL_AMT6": 150_000,
        "PAY_AMT1": 30_000,
        "PAY_AMT2": 25_000,
        "PAY_AMT3": 20_000,
        "PAY_AMT4": 15_000,
        "PAY_AMT5": 10_000,
        "PAY_AMT6": 8_000,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_artifact() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["source_experiment"] != "exp_039":
        raise ValueError("Frozen model manifest does not identify exp_039")
    if manifest["train_py_sha256"] != EXPECTED_TRAIN_SHA256:
        raise ValueError("Frozen manifest has an unexpected train.py hash")
    if _sha256(ROOT / "train.py") != EXPECTED_TRAIN_SHA256:
        raise ValueError("Frozen train.py hash mismatch")
    if manifest["model_artifact_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("Frozen manifest has an unexpected model artifact hash")
    if _sha256(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise ValueError("Frozen model artifact hash mismatch")
    if manifest["sklearn_version"] != sklearn.__version__:
        raise ValueError("Frozen model sklearn version mismatch")
    if manifest["hidden_test_evaluated"] is not False:
        raise ValueError("Frozen inference artifact unexpectedly references hidden evaluation")
    return manifest


@lru_cache(maxsize=1)
def load_frozen_model():
    verify_frozen_artifact()
    return joblib.load(MODEL_PATH)


def synthetic_portfolio(selected_profile: dict, size: int = 24) -> pd.DataFrame:
    if not 20 <= size <= 50:
        raise ValueError("Synthetic portfolio size must be between 20 and 50")
    rng = np.random.default_rng(42)
    rows = [{**selected_profile, "scenario": "Selected profile"}]
    names = list(SCENARIOS)
    for index in range(1, size):
        name = names[(index - 1) % len(names)]
        row = SCENARIOS[name].copy()
        scale = rng.uniform(0.72, 1.28)
        row["LIMIT_BAL"] = max(10_000, round(row["LIMIT_BAL"] * scale / 1_000) * 1_000)
        for column in [f"BILL_AMT{month}" for month in range(1, 7)]:
            row[column] = max(0, round(row[column] * rng.uniform(0.7, 1.3) / 500) * 500)
        for column in [f"PAY_AMT{month}" for month in range(1, 7)]:
            row[column] = max(0, round(row[column] * rng.uniform(0.65, 1.35) / 500) * 500)
        row["scenario"] = f"S{index:02d} · {name}"
        rows.append(row)
    return pd.DataFrame(rows)


def score_portfolio(profiles: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ALLOWED_MODEL_FEATURES) - set(profiles.columns))
    excluded = sorted(set(EXCLUDED_MODEL_FEATURES) & set(profiles.columns))
    unexpected = sorted(set(profiles.columns) - set(ALLOWED_MODEL_FEATURES) - {"scenario"})
    if missing or excluded or unexpected:
        raise ValueError(
            f"Invalid demo features; missing={missing}, excluded={excluded}, "
            f"unexpected={unexpected}"
        )
    financial = profiles.loc[:, ALLOWED_MODEL_FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(financial.to_numpy()).all():
        raise ValueError("Synthetic profile values must be finite")
    if (financial["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limit must be positive")

    probabilities = load_frozen_model().predict_proba(
        approved_feature_matrix(financial, include_derived=True)
    )[:, 1]
    exposure = financial["BILL_AMT1"].clip(lower=0).to_numpy()
    utilization = exposure / financial["LIMIT_BAL"].clip(lower=1).to_numpy()
    recent_ratio = np.clip(
        financial["PAY_AMT1"].to_numpy() / financial["BILL_AMT1"].clip(lower=1).to_numpy(),
        0,
        1,
    )
    bill_columns = [f"BILL_AMT{month}" for month in range(1, 7)]
    payment_columns = [f"PAY_AMT{month}" for month in range(1, 7)]
    mean_ratio = np.clip(
        financial[payment_columns].to_numpy() / financial[bill_columns].clip(lower=1).to_numpy(),
        0,
        1,
    ).mean(axis=1)
    allocation_score = (
        probabilities * np.power(exposure, 0.60) * (1 + utilization) * (2 - mean_ratio)
    )

    actions = np.full(len(financial), 1, dtype=int)
    actions[exposure < 1_000] = 0
    human_count = int(np.floor(0.35 * len(financial)))
    human_indices = np.argsort(-allocation_score, kind="stable")[:human_count]
    actions[human_indices] = 3
    ranking = np.empty(len(financial), dtype=int)
    ranking[np.argsort(-allocation_score, kind="stable")] = np.arange(1, len(financial) + 1)

    return pd.DataFrame(
        {
            "scenario": profiles.get(
                "scenario", pd.Series([f"S{i:02d}" for i in range(len(financial))])
            ),
            "predicted_probability": probabilities,
            "exposure": exposure,
            "utilization": utilization,
            "recent_payment_to_bill_ratio": recent_ratio,
            "mean_payment_to_bill_ratio_6m": mean_ratio,
            "allocation_score": allocation_score,
            "rank": ranking,
            "inside_top_35": ranking <= human_count,
            "action": [ACTION_NAMES[action] for action in actions],
        },
        index=profiles.index,
    )


def risk_label(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.45:
        return "Moderate"
    if probability < 0.70:
        return "High"
    return "Very high"
