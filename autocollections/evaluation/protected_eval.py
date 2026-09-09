import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from autocollections.data.loader import load_dataset
from autocollections.data.schema import TARGET
from autocollections.evaluation.feasibility import rank_with_human_capacity
from autocollections.evaluation.metrics import risk_metric_report
from autocollections.features.approved_features import approved_feature_view
from autocollections.simulator.business_simulator import ACTION_NAMES
from autocollections.simulator.business_simulator_v2 import simulate_policy_v2
from autocollections.utils.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_VERSION = "phase5_evaluator_v1"
DISQUALIFIED_SCORE = -1e18
STRONG_BASELINE_ID = "top35_human_else_digital"
STRONG_BASELINE_UTILITY_TOTAL = 3_623_150.1362439794
STRONG_BASELINE_UTILITY_PER_ACCOUNT = 805.1444747208843
EXPECTED_DATASET_SHA256 = "63e0f53f2617f0d5aaef6202779b78d123dc2c8d1d5cdfffa5a7d4ca20823445"
EXPECTED_SPLIT_SHA256 = "6d8f4e9a9355cb49351af5fb83d6a6a938538d47c084fd039358cbed9ae0e0ec"
EXPECTED_PROTECTED_HASHES = {
    "simulator_config_sha256": "73ead1396946f06fb7bc8679a2b59254ba577cc7f5678f3940be1d92d9d069d5",
    "simulator_implementation_sha256": "f9e609fdbb73ec26889882e44c5bccd1d19083d76d2cf82188414d5eaf02f5a0",
    "feasibility_implementation_sha256": "490a6ef3cc97094f2759d11bf68e05e4a23ab025ebb9c055a14a38151c1e740b",
    "strong_baseline_artifact_sha256": "463363fda5a9cd88de59ac5c862fdc6dace1782c95240f6b23522e958029b5ec",
}


def evaluator_identity() -> dict[str, Any]:
    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    actual_hashes = {
        "simulator_config_sha256": sha256_file(ROOT / "configs/business_simulation_v2.yaml"),
        "simulator_implementation_sha256": sha256_file(
            ROOT / "autocollections/simulator/business_simulator_v2.py"
        ),
        "feasibility_implementation_sha256": sha256_file(
            ROOT / "autocollections/evaluation/feasibility.py"
        ),
        "strong_baseline_artifact_sha256": sha256_file(ROOT / "reports/feasibility_audit_v2.json"),
    }
    if metadata["data_sha256"] != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset does not match the frozen evaluator identity")
    if manifest["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Split does not match the frozen evaluator identity")
    if actual_hashes != EXPECTED_PROTECTED_HASHES:
        raise ValueError("Protected simulator or feasibility inputs changed")
    baseline = json.loads((ROOT / "reports/feasibility_audit_v2.json").read_text())
    measured_baseline = baseline["base_policy_results"][STRONG_BASELINE_ID]["net_business_utility"]
    if not np.isclose(measured_baseline, STRONG_BASELINE_UTILITY_TOTAL, atol=1e-9):
        raise ValueError("Strong baseline utility does not match the frozen evaluator")

    identity = {
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_sha256": metadata["data_sha256"],
        "split_sha256": manifest["split_sha256"],
        **actual_hashes,
        "strong_baseline_id": STRONG_BASELINE_ID,
        "strong_baseline_utility_total": STRONG_BASELINE_UTILITY_TOTAL,
        "strong_baseline_utility_per_account": STRONG_BASELINE_UTILITY_PER_ACCOUNT,
        "row_count": len(frame),
    }
    return {**identity, "evaluator_sha256": sha256_json(identity)}


def load_research_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    evaluator_identity()
    frame, _ = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    train_ids = manifest["splits"]["train"]
    validation_ids = np.asarray(manifest["splits"]["validation"], dtype=int)
    return (
        approved_feature_view(frame.iloc[train_ids]).reset_index(drop=True),
        frame.iloc[train_ids][TARGET].reset_index(drop=True),
        approved_feature_view(frame.iloc[validation_ids]).reset_index(drop=True),
        validation_ids,
    )


def validate_candidate_output(
    candidate: pd.DataFrame,
    expected_sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_columns = ["sample_id", "predicted_probability", "action"]
    if candidate.columns.tolist() != expected_columns:
        raise ValueError(f"Candidate columns must be exactly {expected_columns}")
    if len(candidate) != len(expected_sample_ids):
        raise ValueError("Candidate row count does not match validation rows")
    sample_ids = pd.to_numeric(candidate["sample_id"], errors="raise").to_numpy()
    if not np.array_equal(sample_ids, expected_sample_ids):
        raise ValueError("Candidate sample IDs or row order do not match validation")

    probabilities = pd.to_numeric(candidate["predicted_probability"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Candidate probabilities must be finite and within [0, 1]")
    numeric_actions = pd.to_numeric(candidate["action"], errors="raise").to_numpy()
    if not np.equal(numeric_actions, np.floor(numeric_actions)).all():
        raise ValueError("Candidate actions must be integers")
    actions = numeric_actions.astype(int)
    if not set(np.unique(actions)).issubset(ACTION_NAMES):
        raise ValueError("Candidate actions must be one of 0, 1, 2, or 3")
    return probabilities, actions


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNCOMMITTED"


def evaluate_candidate(
    candidate: pd.DataFrame,
    *,
    experiment_id: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    identity = evaluator_identity()
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    assumptions = yaml.safe_load((ROOT / "configs/business_simulation_v2.yaml").read_text())
    frame, _ = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    validation_ids = np.asarray(manifest["splits"]["validation"], dtype=int)
    validation = frame.iloc[validation_ids]
    probabilities, actions = validate_candidate_output(candidate, validation_ids)

    simulation = simulate_policy_v2(
        approved_feature_view(validation),
        validation[TARGET],
        actions,
        assumptions,
        max_human_escalation_rate=benchmark["policy_constraints"]["max_human_escalation_rate"],
    )
    _, _, feasible_ranking = rank_with_human_capacity(
        {"candidate": simulation},
        max_human_rate=benchmark["policy_constraints"]["max_human_escalation_rate"],
    )
    feasible = bool(feasible_ranking)
    risk_metrics = risk_metric_report(validation[TARGET], probabilities)
    improvement_total = simulation["net_business_utility"] - STRONG_BASELINE_UTILITY_TOTAL
    improvement_per_account = (
        simulation["average_utility_per_account"] - STRONG_BASELINE_UTILITY_PER_ACCOUNT
    )
    beats_baseline = feasible and improvement_total > 1e-9
    over_treatment_penalty = (
        simulation["customer_experience_penalty_total"]
        + simulation["unnecessary_escalation_penalty_total"]
    )
    result = {
        "experiment_id": experiment_id,
        "status": "PASS" if feasible else "INELIGIBLE",
        "primary_score": (
            simulation["average_utility_per_account"] if feasible else DISQUALIFIED_SCORE
        ),
        "protected_business_utility": simulation["net_business_utility"],
        "feasible": feasible,
        "PRIMARY_SCORE_ELIGIBLE": feasible,
        "KEEP_ELIGIBLE": beats_baseline,
        "beats_strong_baseline": beats_baseline,
        "baseline_improvement_total": improvement_total,
        "baseline_improvement_per_account": improvement_per_account,
        "strong_baseline": {
            "id": STRONG_BASELINE_ID,
            "utility_total": STRONG_BASELINE_UTILITY_TOTAL,
            "utility_per_account": STRONG_BASELINE_UTILITY_PER_ACCOUNT,
        },
        "simulated_recovery_benefit_total": simulation["simulated_recovery_benefit_total"],
        "treatment_cost_total": simulation["treatment_cost_total"],
        "over_treatment_penalty_total": over_treatment_penalty,
        "missed_opportunity_penalty_total": simulation["missed_opportunity_penalty_total"],
        "policy_violation_penalty_total": simulation["policy_violation_penalty_total"],
        "net_business_utility": simulation["net_business_utility"],
        "utility_per_account": simulation["average_utility_per_account"],
        "human_escalation_rate": simulation["human_escalation_rate"],
        "action_counts": {
            name: item["count"] for name, item in simulation["action_distribution"].items()
        },
        "roc_auc": risk_metrics["roc_auc"],
        "pr_auc": risk_metrics["pr_auc"],
        "brier_score": risk_metrics["brier_score"],
        "runtime_seconds": float(runtime_seconds),
        "source_commit": _source_commit(),
        "evaluator_version": identity["evaluator_version"],
        "evaluator_sha256": identity["evaluator_sha256"],
        "dataset_sha256": identity["dataset_sha256"],
        "split_sha256": identity["split_sha256"],
        "evaluation_split": "validation",
        "hidden_test_evaluated": False,
    }
    output = ROOT / "reports/experiments/latest_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
