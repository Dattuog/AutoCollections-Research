import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocollections.data.loader import load_dataset
from autocollections.data.schema import TARGET
from autocollections.evaluation.metrics import risk_metric_report
from autocollections.evaluation.protected_eval import evaluator_identity
from autocollections.features.approved_features import (
    approved_feature_matrix,
    approved_feature_view,
)
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
)
from autocollections.simulator.business_simulator_v2 import (
    counterfactual_action_utilities_v2,
    simulate_policy_v2,
)
from autocollections.utils.hashing import sha256_file
from scripts.check_protected import MANIFEST_PATH, protected_hashes, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "reports/final_selection_manifest.json"
START_MARKER_PATH = ROOT / "reports/final_hidden_evaluation_started.json"
OUTPUT_PATH = ROOT / "reports/final_hidden_evaluation.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_selected_candidate(selection: dict) -> None:
    if selection["selected_experiment"] != "exp_039":
        raise ValueError("Final selection is not exp_039")
    if selection["research_experiment_count"] != 50:
        raise ValueError("Final selection does not freeze exactly 50 experiments")
    if sha256_file(ROOT / "train.py") != selection["selected_train_sha256"]:
        raise ValueError("Current train.py does not match the selected candidate")
    if _git("rev-list", "-n", "1", "phase6-final") != selection["selected_commit"]:
        raise ValueError("phase6-final does not identify the selected commit")
    if sum(1 for _ in (ROOT / "results.tsv").open()) - 1 != 50:
        raise ValueError("results.tsv does not contain exactly 50 experiments")


def mark_hidden_evaluation_started(
    selection_path: Path = SELECTION_PATH,
    marker_path: Path = START_MARKER_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    if not selection_path.is_file():
        raise ValueError("Final selection manifest must exist before hidden evaluation")
    if marker_path.exists() or output_path.exists():
        raise ValueError("Hidden evaluation has already started or completed")
    marker = {
        "started_at": datetime.now(UTC).isoformat(),
        "selection_manifest_sha256": sha256_file(selection_path),
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    if selection_path.stat().st_mtime_ns >= marker_path.stat().st_mtime_ns:
        raise ValueError("Selection manifest was not written before hidden evaluation")
    return marker


def selected_candidate_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = approved_feature_matrix(X_train, include_derived=True)
    eval_features = approved_feature_matrix(X_eval, include_derived=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=150,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_features, y_train)
    probabilities = model.predict_proba(eval_features)[:, 1]

    actions = np.full(len(probabilities), DIGITAL_REMINDER, dtype=int)
    exposure = X_eval["BILL_AMT1"].clip(lower=0).to_numpy()
    actions[exposure < 1_000] = NO_CONTACT
    utilization = exposure / X_eval["LIMIT_BAL"].clip(lower=1).to_numpy()
    bill_columns = [f"BILL_AMT{month}" for month in range(1, 7)]
    payment_columns = [f"PAY_AMT{month}" for month in range(1, 7)]
    mean_payment_ratio = np.clip(
        X_eval[payment_columns].to_numpy() / X_eval[bill_columns].clip(lower=1).to_numpy(),
        0,
        1,
    ).mean(axis=1)
    allocation_score = (
        probabilities * np.power(exposure, 0.60) * (1 + utilization) * (2 - mean_payment_ratio)
    )
    human_count = int(np.floor(0.35 * len(probabilities)))
    actions[np.argsort(-allocation_score, kind="stable")[:human_count]] = HUMAN_ESCALATION
    return probabilities, actions


def b4_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = approved_feature_matrix(X_train, include_derived=True)
    eval_features = approved_feature_matrix(X_eval, include_derived=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=42)),
        ]
    )
    model.fit(train_features, y_train)
    probabilities = model.predict_proba(eval_features)[:, 1]
    actions = np.full(len(probabilities), DIGITAL_REMINDER, dtype=int)
    human_count = int(np.floor(0.35 * len(probabilities)))
    actions[np.argsort(-probabilities, kind="stable")[:human_count]] = HUMAN_ESCALATION
    return probabilities, actions


def paired_bootstrap_interval(
    selected_utility: np.ndarray,
    baseline_utility: np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 5_000,
) -> dict:
    difference = np.asarray(selected_utility) - np.asarray(baseline_utility)
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(resamples)
        ]
    )
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "method": "paired account-level bootstrap percentile interval",
        "seed": seed,
        "resamples": resamples,
        "mean_difference": float(difference.mean()),
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def generalization_category(validation_gain: float, hidden_gain: float) -> str:
    if hidden_gain < 0:
        return "negative generalization"
    retained = hidden_gain / validation_gain if validation_gain > 0 else 0.0
    if retained >= 0.75:
        return "strong generalization"
    if retained >= 0.25:
        return "partial generalization"
    return "little/no generalization"


def _policy_report(
    features: pd.DataFrame,
    target: pd.Series,
    probabilities: np.ndarray,
    actions: np.ndarray,
    assumptions: dict,
    max_human_rate: float,
) -> dict:
    business = simulate_policy_v2(
        features,
        target,
        actions,
        assumptions,
        max_human_escalation_rate=max_human_rate,
    )
    metrics = risk_metric_report(target, probabilities)
    return {
        **business,
        "feasible": business["human_escalation_rate"] <= max_human_rate,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier_score": metrics["brier_score"],
        "predicted_risk_mean": metrics["predicted_probability_distribution"]["mean"],
    }


def main() -> None:
    if _git("branch", "--show-current") != "autoresearch":
        raise ValueError("Final evaluation requires branch autoresearch")
    selection = json.loads(SELECTION_PATH.read_text())
    verify_selected_candidate(selection)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    if any(verify_manifest(manifest).values()):
        raise ValueError("Protected integrity failed before final evaluation")
    before_hashes = protected_hashes()
    train_hash_before = sha256_file(ROOT / "train.py")
    evaluation_commit = _git("rev-parse", "HEAD")
    marker = mark_hidden_evaluation_started()

    identity = evaluator_identity()
    if identity["evaluator_sha256"] != selection["evaluator_sha256"]:
        raise ValueError("Evaluator identity differs from final selection")
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    assumptions = yaml.safe_load((ROOT / "configs/business_simulation_v2.yaml").read_text())
    frame, _ = load_dataset()
    split_manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    train_ids = split_manifest["splits"]["train"]
    validation_ids = split_manifest["splits"]["validation"]
    hidden_ids = split_manifest["splits"]["hidden_test"]
    X_train = approved_feature_view(frame.iloc[train_ids]).reset_index(drop=True)
    y_train = frame.iloc[train_ids][TARGET].reset_index(drop=True)
    X_validation = approved_feature_view(frame.iloc[validation_ids]).reset_index(drop=True)
    X_hidden = approved_feature_view(frame.iloc[hidden_ids]).reset_index(drop=True)

    _, validation_actions = selected_candidate_predictions(X_train, y_train, X_validation)
    validation_business = simulate_policy_v2(
        X_validation,
        frame.iloc[validation_ids][TARGET].reset_index(drop=True),
        validation_actions,
        assumptions,
        max_human_escalation_rate=benchmark["policy_constraints"]["max_human_escalation_rate"],
    )
    if not np.isclose(
        validation_business["average_utility_per_account"],
        selection["selected_validation_score"],
        atol=1e-9,
    ):
        raise ValueError("Final evaluator does not reproduce the selected validation candidate")

    hidden_target = frame.iloc[hidden_ids][TARGET].reset_index(drop=True)
    selected_probabilities, selected_actions = selected_candidate_predictions(
        X_train, y_train, X_hidden
    )
    b4_probabilities, b4_actions = b4_predictions(X_train, y_train, X_hidden)
    max_human_rate = benchmark["policy_constraints"]["max_human_escalation_rate"]
    selected = _policy_report(
        X_hidden,
        hidden_target,
        selected_probabilities,
        selected_actions,
        assumptions,
        max_human_rate,
    )
    b4 = _policy_report(
        X_hidden,
        hidden_target,
        b4_probabilities,
        b4_actions,
        assumptions,
        max_human_rate,
    )
    action_utilities = counterfactual_action_utilities_v2(
        X_hidden, hidden_target, assumptions
    ).to_numpy()
    rows = np.arange(len(hidden_target))
    selected_row_utility = action_utilities[rows, selected_actions]
    b4_row_utility = action_utilities[rows, b4_actions]
    if not np.isclose(selected_row_utility.sum(), selected["net_business_utility"]):
        raise ValueError("Selected account utilities do not match aggregate utility")
    if not np.isclose(b4_row_utility.sum(), b4["net_business_utility"]):
        raise ValueError("B4 account utilities do not match aggregate utility")

    validation = selection["selected_validation_metrics"]
    hidden_improvement = selected["net_business_utility"] - b4["net_business_utility"]
    hidden_improvement_per_account = (
        selected["average_utility_per_account"] - b4["average_utility_per_account"]
    )
    validation_gain = (
        validation["utility_per_account"] - selection["strong_baseline_validation_score"]
    )
    category = generalization_category(validation_gain, hidden_improvement_per_account)
    result = {
        "artifact_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "selection_manifest_sha256": sha256_file(SELECTION_PATH),
        "selection_created_at": selection["created_at"],
        "hidden_evaluation_started_at": marker["started_at"],
        "selected_experiment": selection["selected_experiment"],
        "selected_commit": selection["selected_commit"],
        "evaluation_commit": evaluation_commit,
        "selected_train_sha256": train_hash_before,
        "evaluator_version": identity["evaluator_version"],
        "evaluator_sha256": identity["evaluator_sha256"],
        "dataset_sha256": identity["dataset_sha256"],
        "split_sha256": identity["split_sha256"],
        "evaluation_split": "hidden_test",
        "hidden_target_prevalence": float(hidden_target.mean()),
        "selected_candidate": selected,
        "strong_baseline_b4": b4,
        "comparison": {
            "hidden_utility_improvement_over_b4": hidden_improvement,
            "hidden_utility_improvement_per_account": hidden_improvement_per_account,
            "hidden_percentage_improvement_over_b4": (
                hidden_improvement / b4["net_business_utility"] * 100
            ),
            "validation_to_hidden_utility_per_account_change": (
                selected["average_utility_per_account"] - validation["utility_per_account"]
            ),
            "validation_to_hidden_roc_auc_change": selected["roc_auc"] - validation["roc_auc"],
            "validation_to_hidden_pr_auc_change": selected["pr_auc"] - validation["pr_auc"],
            "validation_to_hidden_brier_change": selected["brier_score"]
            - validation["brier_score"],
            "validation_gain_per_account_over_b4": validation_gain,
            "hidden_gain_retention_ratio": hidden_improvement_per_account / validation_gain,
        },
        "paired_utility_difference_confidence_interval": paired_bootstrap_interval(
            selected_row_utility, b4_row_utility
        ),
        "generalization_assessment": category,
        "PAYMENT_PLAN_HIDDEN_USAGE_ZERO": selected["action_distribution"]["PAYMENT_PLAN_REVIEW"][
            "count"
        ]
        == 0,
        "protected_integrity_before": "PASS",
        "normal_research_path_unchanged": True,
        "post_hidden_tuning_performed": False,
        "disclaimer": (
            "All business values are deterministic simulator outputs in simulated_inr_units, "
            "not observed bank savings or causal treatment effects."
        ),
    }
    if manifest_bytes != MANIFEST_PATH.read_bytes() or before_hashes != protected_hashes():
        raise ValueError("Protected files changed during final evaluation")
    if sha256_file(ROOT / "train.py") != train_hash_before:
        raise ValueError("Selected train.py changed during final evaluation")
    if _git("rev-parse", "HEAD") != evaluation_commit:
        raise ValueError("Git HEAD changed during final evaluation")
    result["protected_integrity_after"] = "PASS"
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("Phase 7 one-time hidden-test evaluation")
    print(f"selected_commit: {selection['selected_commit']}")
    print(f"evaluation_commit: {evaluation_commit}")
    print(f"hidden_prevalence: {hidden_target.mean():.6f}")
    print("policy\tutility\tutility/account\thuman_rate\troc_auc\tpr_auc\tbrier")
    for name, report in (("exp_039", selected), ("B4", b4)):
        print(
            f"{name}\t{report['net_business_utility']:.2f}\t"
            f"{report['average_utility_per_account']:.6f}\t"
            f"{report['human_escalation_rate']:.6f}\t{report['roc_auc']:.6f}\t"
            f"{report['pr_auc']:.6f}\t{report['brier_score']:.6f}"
        )
    print(f"hidden_improvement: {hidden_improvement:.2f}")
    print(f"generalization: {category}")
    print(f"artifact_sha256: {sha256_file(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
