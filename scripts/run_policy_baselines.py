import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocollections.data.loader import load_dataset
from autocollections.data.schema import TARGET
from autocollections.evaluation.baselines import (
    choose_risk_based_actions,
    fit_canonical_risk_model,
)
from autocollections.features.approved_features import approved_feature_view
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
    simulate_policy,
)
from autocollections.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports/policy_baselines.json"


def main() -> None:
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    assumptions = yaml.safe_load((ROOT / "configs/business_simulation.yaml").read_text())
    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    if manifest["split_sha256"] != benchmark["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")

    risk_source = ROOT / "reports/baselines.json"
    risk_report = json.loads(risk_source.read_text())
    if risk_report["selected_model"] != "logistic_derived":
        raise ValueError("Phase 3 source artifact does not select logistic_derived")
    if risk_report["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Risk source artifact and policy evaluation use different splits")

    _, probabilities = fit_canonical_risk_model(frame, manifest, seed=benchmark["seed"])
    validation_ids = manifest["splits"]["validation"]
    validation = frame.iloc[validation_ids]
    features = approved_feature_view(validation)
    target = validation[TARGET]
    risk_actions, risk_bands = choose_risk_based_actions(
        probabilities, assumptions["policy_thresholds"]
    )
    policies = {
        "always_no_contact": np.full(len(validation), NO_CONTACT),
        "always_digital_reminder": np.full(len(validation), DIGITAL_REMINDER),
        "always_payment_plan_review": np.full(len(validation), PAYMENT_PLAN_REVIEW),
        "always_human_escalation": np.full(len(validation), HUMAN_ESCALATION),
        "risk_based_four_action": risk_actions,
    }
    policy_results = {
        name: simulate_policy(
            features,
            target,
            actions,
            probabilities,
            assumptions,
            max_human_escalation_rate=benchmark["policy_constraints"]["max_human_escalation_rate"],
            risk_bands=risk_bands if name == "risk_based_four_action" else None,
        )
        for name, actions in policies.items()
    }
    valid_policies = {
        name: result
        for name, result in policy_results.items()
        if result["policy_violation_count"] == 0
    }
    highest_utility_policy = max(
        valid_policies,
        key=lambda name: valid_policies[name]["average_utility_per_account"],
    )
    report = {
        "artifact_version": 1,
        "simulation_version": assumptions["simulation_version"],
        "disclaimer": (
            "All recovery and utility values are deterministic research simulation outputs, "
            "not observed bank outcomes, revenue, savings, or causal treatment effects."
        ),
        "dataset_sha256": metadata["data_sha256"],
        "split_sha256": manifest["split_sha256"],
        "evaluation_split": "validation",
        "hidden_test_evaluated": False,
        "risk_model": {
            "id": "logistic_derived",
            "source_artifact": "reports/baselines.json",
            "source_artifact_sha256": sha256_file(risk_source),
        },
        "policy_thresholds": assumptions["policy_thresholds"],
        "simulator_assumptions": assumptions,
        "phase4_baseline_policy": "risk_based_four_action",
        "highest_utility_policy_without_violations": highest_utility_policy,
        "policies": policy_results,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("Phase 4 simulated treatment-policy baselines")
    print("Evaluation: validation only; hidden test NOT EVALUATED")
    print(
        "policy\trecovery\tcost\texperience\tescalation\tpolicy_penalty\tutility\tutility/account"
    )
    for name, result in policy_results.items():
        print(
            f"{name}\t{result['simulated_recovery_benefit_total']:.2f}\t"
            f"{result['treatment_cost_total']:.2f}\t"
            f"{result['customer_experience_penalty_total']:.2f}\t"
            f"{result['unnecessary_escalation_penalty_total']:.2f}\t"
            f"{result['policy_violation_penalty_total']:.2f}\t"
            f"{result['net_business_utility']:.2f}\t"
            f"{result['average_utility_per_account']:.4f}"
        )
    print("Phase 4 baseline policy: risk_based_four_action")
    print(f"Highest valid simulated utility: {highest_utility_policy}")
    print(f"Artifact SHA-256: {sha256_file(OUTPUT_PATH)}")
    print("Wrote reports/policy_baselines.json")


if __name__ == "__main__":
    main()
