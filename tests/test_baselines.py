import numpy as np
import pandas as pd

from autocollections.data.schema import (
    ALL_FEATURES,
    ALLOWED_MODEL_FEATURES,
    EXCLUDED_MODEL_FEATURES,
    TARGET,
)
from autocollections.data.splits import create_split_manifest
from autocollections.evaluation.baselines import choose_risk_based_actions, evaluate_baselines
from autocollections.features.approved_features import approved_feature_view
from autocollections.utils.hashing import sha256_dataframe

FRACTIONS = {"train": 0.70, "validation": 0.15, "hidden_test": 0.15}
MODEL_NAMES = {
    "dummy_prior",
    "logistic_raw",
    "logistic_derived",
    "logistic_derived_balanced",
    "logistic_derived_balanced_sigmoid",
}
METRIC_KEYS = {
    "roc_auc",
    "pr_auc",
    "brier_score",
    "log_loss",
    "precision_at_0_5",
    "recall_at_0_5",
    "f1_at_0_5",
    "confusion_matrix_at_0_5",
    "predicted_probability_distribution",
    "calibration",
}


def synthetic_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(index=range(rows), columns=ALL_FEATURES, dtype=float)
    frame["LIMIT_BAL"] = rng.integers(20_000, 500_000, rows)
    frame["SEX"] = rng.integers(1, 3, rows)
    frame["EDUCATION"] = rng.integers(1, 5, rows)
    frame["MARRIAGE"] = rng.integers(1, 4, rows)
    frame["AGE"] = rng.integers(21, 75, rows)
    for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        frame[column] = rng.integers(-2, 5, rows)
    for month in range(1, 7):
        frame[f"BILL_AMT{month}"] = rng.integers(0, 300_000, rows)
        frame[f"PAY_AMT{month}"] = rng.integers(0, 80_000, rows)
    logit = (
        -2.4 + 0.55 * frame["PAY_0"].clip(lower=0) + 0.8 * frame["BILL_AMT1"] / frame["LIMIT_BAL"]
    )
    probability = 1 / (1 + np.exp(-logit))
    frame[TARGET] = rng.binomial(1, probability).astype("int8")
    return frame.loc[:, [*ALL_FEATURES, TARGET]]


def fixture_manifest(frame: pd.DataFrame) -> dict:
    return create_split_manifest(
        frame[TARGET],
        approved_feature_view(frame),
        seed=42,
        fractions=FRACTIONS,
        dataset_sha256=sha256_dataframe(frame),
    )


def test_baseline_schema_feature_enforcement_and_training_only_preprocessing() -> None:
    frame = synthetic_frame()
    manifest = fixture_manifest(frame)
    train_ids = manifest["splits"]["train"]
    validation_ids = manifest["splits"]["validation"]
    frame.loc[train_ids[0], "LIMIT_BAL"] = np.nan
    frame.loc[validation_ids, "LIMIT_BAL"] = 1_000_000_000
    expected_median = float(frame.loc[train_ids, "LIMIT_BAL"].median())

    report, models, _ = evaluate_baselines(frame, manifest, seed=42, dataset_sha256="fixture")
    raw_pipeline = models["logistic_raw"]
    imputer = raw_pipeline.named_steps["imputer"]
    scaler = raw_pipeline.named_steps["scaler"]
    imputed_train = frame.loc[train_ids, ALLOWED_MODEL_FEATURES].fillna(
        {"LIMIT_BAL": expected_median}
    )

    assert imputer.statistics_[0] == expected_median
    assert np.isclose(scaler.mean_[0], imputed_train["LIMIT_BAL"].mean())
    assert set(report["models"]) == MODEL_NAMES
    assert report["hidden_test_evaluated"] is False
    assert report["evaluation_split"] == "validation"
    assert not set(EXCLUDED_MODEL_FEATURES) & set(report["feature_sets"]["raw"])
    for result in report["models"].values():
        assert METRIC_KEYS <= set(result)
        assert np.asarray(result["confusion_matrix_at_0_5"]).shape == (2, 2)
        assert result["calibration"]["bins"]


def test_baselines_are_deterministic_and_ignore_hidden_test_rows() -> None:
    frame = synthetic_frame()
    manifest = fixture_manifest(frame)
    first, _, first_probabilities = evaluate_baselines(
        frame, manifest, seed=42, dataset_sha256="fixture"
    )
    second, _, second_probabilities = evaluate_baselines(
        frame, manifest, seed=42, dataset_sha256="fixture"
    )

    changed_hidden = frame.copy()
    hidden_ids = manifest["splits"]["hidden_test"]
    changed_hidden.loc[hidden_ids, ALLOWED_MODEL_FEATURES] += 9_000_000
    changed_hidden.loc[hidden_ids, TARGET] = 1 - changed_hidden.loc[hidden_ids, TARGET]
    after_hidden_change, _, hidden_change_probabilities = evaluate_baselines(
        changed_hidden, manifest, seed=42, dataset_sha256="fixture"
    )

    assert first == second == after_hidden_change
    for name in MODEL_NAMES:
        np.testing.assert_array_equal(first_probabilities[name], second_probabilities[name])
        np.testing.assert_array_equal(first_probabilities[name], hidden_change_probabilities[name])


def test_risk_policy_uses_only_probabilities_and_fixed_bands() -> None:
    probabilities = np.array([0.0, 0.1999, 0.2, 0.4499, 0.45, 0.6999, 0.7, 1.0])
    actions, bands = choose_risk_based_actions(
        probabilities,
        {
            "no_contact_upper": 0.2,
            "digital_reminder_upper": 0.45,
            "payment_plan_review_upper": 0.7,
        },
    )

    assert actions.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert bands.tolist() == [
        "lowest",
        "lowest",
        "low_moderate",
        "low_moderate",
        "higher",
        "higher",
        "highest",
        "highest",
    ]
