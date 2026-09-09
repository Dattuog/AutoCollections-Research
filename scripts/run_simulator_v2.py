import json
import sys
from copy import deepcopy
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
    dominance_diagnostic,
)
from autocollections.simulator.business_simulator_v2 import (
    PROXY_DEFINITIONS,
    SUITABILITY_DEFINITIONS,
    action_suitability_v2,
    behavioral_proxies_v2,
    protected_oracle_v2,
    simulate_policy_v2,
)
from autocollections.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "reports/policy_baselines_v2.json"
SENSITIVITY_PATH = ROOT / "reports/simulator_sensitivity_v2.json"


def sensitivity_scenarios(base: dict) -> dict[str, dict]:
    scenarios = {"base": deepcopy(base)}
    for name, scale in (("treatment_friction_low", 0.5), ("treatment_friction_high", 2.0)):
        scenario = deepcopy(base)
        for action in ("DIGITAL_REMINDER", "PAYMENT_PLAN_REVIEW", "HUMAN_ESCALATION"):
            scenario["costs"][action] *= scale
        for penalty in scenario["experience_penalties"]:
            scenario["experience_penalties"][penalty] *= scale
        scenarios[name] = scenario
    for name, scale in (("treatment_effect_low", 0.5), ("treatment_effect_high", 1.5)):
        scenario = deepcopy(base)
        for action in scenario["action_base_effects"]:
            scenario["action_base_effects"][action] *= scale
        scenarios[name] = scenario
    for name, rate in (("missed_opportunity_low", 0.005), ("missed_opportunity_high", 0.03)):
        scenario = deepcopy(base)
        scenario["missed_opportunity"]["no_contact_recoverable_exposure_rate"] = rate
        scenarios[name] = scenario
    return scenarios


def evaluate_scenario(
    features,
    target,
    policies: dict[str, np.ndarray],
    assumptions: dict,
    max_human_rate: float,
) -> dict:
    policy_results = {
        name: simulate_policy_v2(
            features,
            target,
            actions,
            assumptions,
            max_human_escalation_rate=max_human_rate,
        )
        for name, actions in policies.items()
    }
    _, oracle = protected_oracle_v2(
        features,
        target,
        assumptions,
        max_human_escalation_rate=max_human_rate,
    )
    ranking = sorted(
        policy_results,
        key=lambda name: policy_results[name]["net_business_utility"],
        reverse=True,
    )
    valid_ranking = [
        name for name in ranking if policy_results[name]["policy_violation_count"] == 0
    ]
    return {
        "ranking": ranking,
        "highest_utility_policy": ranking[0],
        "highest_utility_policy_without_violations": valid_ranking[0],
        "policy_utilities": {
            name: policy_results[name]["net_business_utility"] for name in ranking
        },
        "oracle_utility": oracle["net_business_utility"],
        "policies": policy_results,
    }


def main() -> None:
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    v1_assumptions = yaml.safe_load((ROOT / "configs/business_simulation.yaml").read_text())
    assumptions = yaml.safe_load((ROOT / "configs/business_simulation_v2.yaml").read_text())
    if assumptions["policy_thresholds"] != v1_assumptions["policy_thresholds"]:
        raise ValueError("phase4_v2 changed the accepted risk-policy thresholds")

    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    if manifest["split_sha256"] != benchmark["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")
    risk_source = ROOT / "reports/baselines.json"
    risk_report = json.loads(risk_source.read_text())
    if risk_report["selected_model"] != "logistic_derived":
        raise ValueError("Phase 3 source artifact does not select logistic_derived")
    if risk_report["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Risk source artifact and v2 evaluator use different splits")

    _, probabilities = fit_canonical_risk_model(frame, manifest, seed=benchmark["seed"])
    validation = frame.iloc[manifest["splits"]["validation"]]
    features = approved_feature_view(validation)
    target = validation[TARGET]
    risk_actions, _ = choose_risk_based_actions(probabilities, assumptions["policy_thresholds"])
    policies = {
        "always_no_contact": np.full(len(validation), NO_CONTACT),
        "always_digital_reminder": np.full(len(validation), DIGITAL_REMINDER),
        "always_payment_plan_review": np.full(len(validation), PAYMENT_PLAN_REVIEW),
        "always_human_escalation": np.full(len(validation), HUMAN_ESCALATION),
        "risk_based_four_action": risk_actions,
    }
    max_human_rate = benchmark["policy_constraints"]["max_human_escalation_rate"]
    base = evaluate_scenario(features, target, policies, assumptions, max_human_rate)
    oracle_actions, oracle = protected_oracle_v2(
        features,
        target,
        assumptions,
        max_human_escalation_rate=max_human_rate,
    )
    risk = base["policies"]["risk_based_four_action"]
    proxies = behavioral_proxies_v2(features, assumptions)
    suitability = action_suitability_v2(features, assumptions)
    comparison = {
        "artifact_version": 1,
        "simulation_version": assumptions["simulation_version"],
        "value_units": assumptions["value_units"],
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
        "policy_thresholds_unchanged": assumptions["policy_thresholds"],
        "assumptions": assumptions,
        "behavioral_proxy_definitions": PROXY_DEFINITIONS,
        "action_suitability_definitions": SUITABILITY_DEFINITIONS,
        "behavioral_proxy_summary": proxies.agg(["min", "mean", "max"]).to_dict(),
        "action_suitability_summary": suitability.agg(["min", "mean", "max"]).to_dict(),
        "policies": base["policies"],
        "policy_ranking": base["ranking"],
        "highest_utility_policy": base["highest_utility_policy"],
        "highest_utility_policy_without_violations": base[
            "highest_utility_policy_without_violations"
        ],
        "protected_oracle": oracle,
        "oracle_action_percentages": {
            name: float((oracle_actions == action).mean())
            for action, name in (
                (NO_CONTACT, "NO_CONTACT"),
                (DIGITAL_REMINDER, "DIGITAL_REMINDER"),
                (PAYMENT_PLAN_REVIEW, "PAYMENT_PLAN_REVIEW"),
                (HUMAN_ESCALATION, "HUMAN_ESCALATION"),
            )
        },
        "risk_policy_to_oracle_gap": {
            "absolute": oracle["net_business_utility"] - risk["net_business_utility"],
            "per_account": (oracle["net_business_utility"] - risk["net_business_utility"])
            / len(validation),
        },
    }
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    sensitivity = {}
    dominance_input = {}
    for name, scenario_assumptions in sensitivity_scenarios(assumptions).items():
        result = evaluate_scenario(features, target, policies, scenario_assumptions, max_human_rate)
        sensitivity[name] = {
            "assumption_values": {
                "costs": scenario_assumptions["costs"],
                "experience_penalties": scenario_assumptions["experience_penalties"],
                "action_base_effects": scenario_assumptions["action_base_effects"],
                "missed_opportunity": scenario_assumptions["missed_opportunity"],
            },
            **{key: value for key, value in result.items() if key != "policies"},
        }
        dominance_input[name] = {"policies": result["policies"]}
    diagnostic = dominance_diagnostic(dominance_input)
    meaningful_oracle_actions = sum(
        item["rate"] >= 0.01 for item in oracle["action_distribution"].values()
    )
    freeze_diagnostics = {
        "treatment_effectiveness_is_heterogeneous": bool(
            (suitability.drop(columns="NO_CONTACT").nunique() > 1).all()
        ),
        "oracle_uses_multiple_actions_at_one_percent_or_more": meaningful_oracle_actions >= 2,
        "single_static_action_not_broadly_dominant": diagnostic["most_common_winner_count"]
        < int(np.ceil(0.75 * diagnostic["scenario_count"])),
        "assumptions_are_explainable": True,
    }
    sensitivity_report = {
        "artifact_version": 1,
        "audit": "phase4_6_heterogeneous_simulator_sensitivity",
        "simulation_version": assumptions["simulation_version"],
        "value_units": assumptions["value_units"],
        "disclaimer": comparison["disclaimer"],
        "dataset_sha256": metadata["data_sha256"],
        "split_sha256": manifest["split_sha256"],
        "evaluation_split": "validation",
        "hidden_test_evaluated": False,
        "sensitivity_scenarios": sensitivity,
        "dominance_diagnostic": diagnostic,
        "freeze_diagnostics": {
            **freeze_diagnostics,
            "preliminary_ready_for_phase5": all(freeze_diagnostics.values()),
        },
    }
    SENSITIVITY_PATH.write_text(json.dumps(sensitivity_report, indent=2, sort_keys=True) + "\n")

    print("Phase 4.6 heterogeneous simulator comparison")
    print("Evaluation: validation only; hidden test NOT EVALUATED")
    print("policy\tutility\tutility/account\tviolations")
    for name in base["ranking"]:
        result = base["policies"][name]
        print(
            f"{name}\t{result['net_business_utility']:.2f}\t"
            f"{result['average_utility_per_account']:.4f}\t"
            f"{result['policy_violation_count']}"
        )
    print(
        f"protected_oracle\t{oracle['net_business_utility']:.2f}\t"
        f"{oracle['average_utility_per_account']:.4f}\t{oracle['policy_violation_count']}"
    )
    print("oracle_actions\t" + json.dumps(oracle["action_distribution"], sort_keys=True))
    print("Phase 4.6 sensitivity")
    print("scenario\twinner\thighest_valid\tranking")
    for name, result in sensitivity.items():
        print(
            f"{name}\t{result['highest_utility_policy']}\t"
            f"{result['highest_utility_policy_without_violations']}\t"
            f"{' > '.join(result['ranking'])}"
        )
    print(f"Dominance warnings: {len(diagnostic['warnings'])}")
    print(
        "Preliminary freeze criteria: " + ("PASS" if all(freeze_diagnostics.values()) else "FAIL")
    )
    print(f"Comparison artifact SHA-256: {sha256_file(COMPARISON_PATH)}")
    print(f"Sensitivity artifact SHA-256: {sha256_file(SENSITIVITY_PATH)}")


if __name__ == "__main__":
    main()
