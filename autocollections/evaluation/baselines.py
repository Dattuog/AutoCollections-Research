import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autocollections-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autocollections.data.loader import load_dataset
from autocollections.data.schema import ALLOWED_MODEL_FEATURES, EXCLUDED_MODEL_FEATURES, TARGET
from autocollections.data.splits import validate_split_manifest
from autocollections.evaluation.metrics import risk_metric_report
from autocollections.features.approved_features import (
    DERIVED_MODEL_FEATURES,
    approved_feature_matrix,
    approved_feature_view,
)
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports/baselines.json"
FIGURE_PATH = ROOT / "reports/figures/baseline_diagnostics.png"


def choose_risk_based_actions(
    probabilities: np.ndarray, thresholds: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1]")
    boundaries = np.array(
        [
            thresholds["no_contact_upper"],
            thresholds["digital_reminder_upper"],
            thresholds["payment_plan_review_upper"],
        ]
    )
    if not (0 < boundaries[0] < boundaries[1] < boundaries[2] < 1):
        raise ValueError("Policy thresholds must be strictly increasing within (0, 1)")

    actions = np.array([NO_CONTACT, DIGITAL_REMINDER, PAYMENT_PLAN_REVIEW, HUMAN_ESCALATION])[
        np.digitize(probability, boundaries)
    ]
    bands = np.array(["lowest", "low_moderate", "higher", "highest"])[
        np.digitize(probability, boundaries)
    ]
    return actions, bands


def fit_canonical_risk_model(
    frame: pd.DataFrame, manifest: dict[str, Any], *, seed: int
) -> tuple[Pipeline, np.ndarray]:
    train_ids = manifest["splits"]["train"]
    validation_ids = manifest["splits"]["validation"]
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation rows overlap")

    X_train = approved_feature_matrix(frame.iloc[train_ids], include_derived=True)
    X_validation = approved_feature_matrix(frame.iloc[validation_ids], include_derived=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )
    model.fit(X_train, frame.iloc[train_ids][TARGET])
    return model, model.predict_proba(X_validation)[:, 1]


def evaluate_baselines(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    seed: int,
    dataset_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    train_ids = manifest["splits"]["train"]
    validation_ids = manifest["splits"]["validation"]
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation rows overlap")

    train = frame.iloc[train_ids]
    validation = frame.iloc[validation_ids]
    raw_train = approved_feature_matrix(train, include_derived=False)
    raw_validation = approved_feature_matrix(validation, include_derived=False)
    derived_train = approved_feature_matrix(train, include_derived=True)
    derived_validation = approved_feature_matrix(validation, include_derived=True)
    y_train = train[TARGET]
    y_validation = validation[TARGET]

    canonical_model, canonical_probabilities = fit_canonical_risk_model(frame, manifest, seed=seed)
    models: dict[str, Any] = {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_raw": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2_000, random_state=seed)),
            ]
        ),
        "logistic_derived": canonical_model,
        "logistic_derived_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=seed),
                ),
            ]
        ),
        "logistic_derived_balanced_sigmoid": CalibratedClassifierCV(
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2_000, random_state=seed
                        ),
                    ),
                ]
            ),
            method="sigmoid",
            cv=5,
        ),
    }
    feature_sets = {
        "dummy_prior": (raw_train, raw_validation),
        "logistic_raw": (raw_train, raw_validation),
        "logistic_derived": (derived_train, derived_validation),
        "logistic_derived_balanced": (derived_train, derived_validation),
        "logistic_derived_balanced_sigmoid": (derived_train, derived_validation),
    }
    descriptions = {
        "dummy_prior": {"feature_set": "raw", "class_weight": None, "calibration": None},
        "logistic_raw": {"feature_set": "raw", "class_weight": None, "calibration": None},
        "logistic_derived": {
            "feature_set": "raw_plus_derived",
            "class_weight": None,
            "calibration": None,
        },
        "logistic_derived_balanced": {
            "feature_set": "raw_plus_derived",
            "class_weight": "balanced",
            "calibration": None,
        },
        "logistic_derived_balanced_sigmoid": {
            "feature_set": "raw_plus_derived",
            "class_weight": "balanced",
            "calibration": "sigmoid_cv5",
        },
    }

    results = {}
    probabilities = {}
    for name, model in models.items():
        X_train, X_validation = feature_sets[name]
        if name == "logistic_derived":
            probabilities[name] = canonical_probabilities
        else:
            model.fit(X_train, y_train)
            probabilities[name] = model.predict_proba(X_validation)[:, 1]
        results[name] = {
            **descriptions[name],
            **risk_metric_report(y_validation, probabilities[name]),
        }

    real_models = tuple(name for name in models if name != "dummy_prior")
    selected_model = min(
        real_models,
        key=lambda name: (
            results[name]["brier_score"],
            results[name]["log_loss"],
            -results[name]["roc_auc"],
        ),
    )
    report = {
        "artifact_version": 1,
        "dataset_sha256": dataset_sha256,
        "split_sha256": manifest["split_sha256"],
        "seed": seed,
        "evaluation_split": "validation",
        "hidden_test_evaluated": False,
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "reporting_threshold": 0.5,
        "feature_sets": {
            "raw": list(ALLOWED_MODEL_FEATURES),
            "derived": list(DERIVED_MODEL_FEATURES),
            "excluded": list(EXCLUDED_MODEL_FEATURES),
        },
        "selection_rule": "lowest validation Brier score, then log loss, then ROC-AUC",
        "selected_model": selected_model,
        "models": results,
    }
    return report, models, probabilities


def _write_diagnostic_figure(report: dict[str, Any], probabilities: dict[str, np.ndarray]) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in report["models"].items():
        axes[0].hist(
            probabilities[name], bins=40, density=True, histtype="step", linewidth=1.5, label=name
        )
        bins = result["calibration"]["bins"]
        axes[1].plot(
            [item["mean_predicted_probability"] for item in bins],
            [item["observed_default_rate"] for item in bins],
            marker="o",
            markersize=3,
            label=name,
        )
    axes[0].set(title="Validation probability distributions", xlabel="P(default)", ylabel="Density")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    axes[1].set(
        title="Validation calibration (uniform bins)",
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    validate_split_manifest(manifest, approved_feature_view(frame))
    if manifest["split_sha256"] != config["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")

    report, _, probabilities = evaluate_baselines(
        frame,
        manifest,
        seed=config["seed"],
        dataset_sha256=metadata["data_sha256"],
    )
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_diagnostic_figure(report, probabilities)

    print("Phase 3 default-risk baselines")
    print("Evaluation: validation only; hidden test NOT EVALUATED")
    print("model\troc_auc\tpr_auc\tbrier\tlog_loss\tprecision\trecall\tf1")
    for name, result in report["models"].items():
        print(
            f"{name}\t{result['roc_auc']:.6f}\t{result['pr_auc']:.6f}\t"
            f"{result['brier_score']:.6f}\t{result['log_loss']:.6f}\t"
            f"{result['precision_at_0_5']:.6f}\t{result['recall_at_0_5']:.6f}\t"
            f"{result['f1_at_0_5']:.6f}"
        )
    print(f"Selected baseline: {report['selected_model']}")
    print("Wrote reports/baselines.json and reports/figures/baseline_diagnostics.png")


if __name__ == "__main__":
    main()
