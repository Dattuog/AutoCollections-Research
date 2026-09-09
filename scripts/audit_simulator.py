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
    simulate_policy,
)
from autocollections.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports/simulator_sensitivity.json"


def sensitivity_scenarios(base: dict) -> dict[str, dict]:
    scenarios = {"base": deepcopy(base)}
    changes = (
        ("payment_plan_cost_low", "costs", "PAYMENT_PLAN_REVIEW", 25.0),
        ("payment_plan_cost_high", "costs", "PAYMENT_PLAN_REVIEW", 100.0),
        (
            "payment_plan_penalty_low",
            "experience_penalties",
            "unnecessary_payment_plan_review",
            10.0,
        ),
        (
            "payment_plan_penalty_high",
            "experience_penalties",
            "unnecessary_payment_plan_review",
            75.0,
        ),
        ("digital_cost_low", "costs", "DIGITAL_REMINDER", 2.0),
        ("digital_cost_high", "costs", "DIGITAL_REMINDER", 15.0),
        ("human_cost_low", "costs", "HUMAN_ESCALATION", 75.0),
        ("human_cost_high", "costs", "HUMAN_ESCALATION", 300.0),
        ("missed_opportunity_low", "missed_opportunity", "no_contact_default_rate", 0.005),
        ("missed_opportunity_high", "missed_opportunity", "no_contact_default_rate", 0.03),
    )
    for name, section, key, value in changes:
        scenario = deepcopy(base)
        scenario[section][key] = value
        scenarios[name] = scenario

    for name, scale in (("recovery_bounds_low", 0.5), ("recovery_bounds_high", 1.5)):
        scenario = deepcopy(base)
        scenario["recovery_rate_bounds"] = {
            action: [low * scale, high * scale]
            for action, (low, high) in base["recovery_rate_bounds"].items()
        }
        scenarios[name] = scenario
    return scenarios


def main() -> None:
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    base_assumptions = yaml.safe_load((ROOT / "configs/business_simulation.yaml").read_text())
    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    if manifest["split_sha256"] != benchmark["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")

    risk_source = ROOT / "reports/baselines.json"
    risk_report = json.loads(risk_source.read_text())
    if risk_report["selected_model"] != "logistic_derived":
        raise ValueError("Phase 3 source artifact does not select logistic_derived")
    if risk_report["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Risk source artifact and sensitivity audit use different splits")

    _, probabilities = fit_canonical_risk_model(frame, manifest, seed=benchmark["seed"])
    validation = frame.iloc[manifest["splits"]["validation"]]
    features = approved_feature_view(validation)
    target = validation[TARGET]
    risk_actions, _ = choose_risk_based_actions(
        probabilities, base_assumptions["policy_thresholds"]
    )
    policies = {
        "always_no_contact": np.full(len(validation), NO_CONTACT),
        "always_digital_reminder": np.full(len(validation), DIGITAL_REMINDER),
        "always_payment_plan_review": np.full(len(validation), PAYMENT_PLAN_REVIEW),
        "always_human_escalation": np.full(len(validation), HUMAN_ESCALATION),
        "risk_based_four_action": risk_actions,
    }

    scenarios = {}
    for scenario_name, assumptions in sensitivity_scenarios(base_assumptions).items():
        results = {
            policy_name: simulate_policy(
                features,
                target,
                actions,
                probabilities,
                assumptions,
                max_human_escalation_rate=benchmark["policy_constraints"][
                    "max_human_escalation_rate"
                ],
            )
            for policy_name, actions in policies.items()
        }
        ranking = sorted(
            results, key=lambda name: results[name]["net_business_utility"], reverse=True
        )
        valid_ranking = [name for name in ranking if results[name]["policy_violation_count"] == 0]
        scenarios[scenario_name] = {
            "assumption_values": {
                "payment_plan_treatment_cost": assumptions["costs"]["PAYMENT_PLAN_REVIEW"],
                "payment_plan_over_treatment_penalty": assumptions["experience_penalties"][
                    "unnecessary_payment_plan_review"
                ],
                "digital_treatment_cost": assumptions["costs"]["DIGITAL_REMINDER"],
                "human_escalation_cost": assumptions["costs"]["HUMAN_ESCALATION"],
                "missed_opportunity_rate": assumptions["missed_opportunity"][
                    "no_contact_default_rate"
                ],
                "recovery_rate_bounds": assumptions["recovery_rate_bounds"],
            },
            "ranking": ranking,
            "highest_utility_policy": ranking[0],
            "highest_utility_policy_without_violations": valid_ranking[0],
            "policy_utilities": {name: results[name]["net_business_utility"] for name in ranking},
            "policies": results,
        }

    plan = scenarios["base"]["policies"]["always_payment_plan_review"]
    risk = scenarios["base"]["policies"]["risk_based_four_action"]
    accounts = len(validation)
    absolute = {
        "additional_simulated_recovery": plan["simulated_recovery_benefit_total"]
        - risk["simulated_recovery_benefit_total"],
        "additional_treatment_cost": plan["treatment_cost_total"] - risk["treatment_cost_total"],
        "additional_over_treatment_penalty": (
            plan["customer_experience_penalty_total"]
            + plan["unnecessary_escalation_penalty_total"]
            - risk["customer_experience_penalty_total"]
            - risk["unnecessary_escalation_penalty_total"]
        ),
        "avoided_missed_opportunity_penalty": risk["missed_opportunity_penalty_total"]
        - plan["missed_opportunity_penalty_total"],
        "additional_policy_penalty": plan["policy_violation_penalty_total"]
        - risk["policy_violation_penalty_total"],
        "net_utility_advantage": plan["net_business_utility"] - risk["net_business_utility"],
    }
    decomposition = {
        "comparison": "always_payment_plan_review minus risk_based_four_action",
        "absolute": absolute,
        "per_account": {name: value / accounts for name, value in absolute.items()},
    }
    compact_scenarios = {
        name: {key: value for key, value in scenario.items() if key != "policies"}
        for name, scenario in scenarios.items()
    }
    report = {
        "artifact_version": 1,
        "audit": "phase4_5_simulator_sanity_and_sensitivity",
        "simulation_version": base_assumptions["simulation_version"],
        "value_units": base_assumptions["value_units"],
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
        "policy_thresholds_unchanged": base_assumptions["policy_thresholds"],
        "utility_decomposition": decomposition,
        "sensitivity_scenarios": compact_scenarios,
        "dominance_diagnostic": dominance_diagnostic(scenarios),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("Phase 4.5 simulator sanity and sensitivity audit")
    print("Evaluation: validation only; hidden test NOT EVALUATED")
    print("scenario\twinner\thighest_valid\tranking")
    for name, scenario in compact_scenarios.items():
        print(
            f"{name}\t{scenario['highest_utility_policy']}\t"
            f"{scenario['highest_utility_policy_without_violations']}\t"
            f"{' > '.join(scenario['ranking'])}"
        )
    print(f"Dominance warnings: {len(report['dominance_diagnostic']['warnings'])}")
    print(f"Artifact SHA-256: {sha256_file(OUTPUT_PATH)}")
    print("Wrote reports/simulator_sensitivity.json")


if __name__ == "__main__":
    main()
