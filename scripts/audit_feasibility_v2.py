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
from autocollections.evaluation.feasibility import (
    CAPACITY_POLICIES,
    capacity_saturation_diagnostic,
    rank_with_human_capacity,
    top_risk_human_policy,
)
from autocollections.features.approved_features import approved_feature_view
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
)
from autocollections.simulator.business_simulator_v2 import (
    protected_oracle_v2,
    simulate_policy_v2,
)
from autocollections.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports/feasibility_audit_v2.json"
ACCEPTED_HASHES = {
    "config": "73ead1396946f06fb7bc8679a2b59254ba577cc7f5678f3940be1d92d9d069d5",
    "simulator": "f9e609fdbb73ec26889882e44c5bccd1d19083d76d2cf82188414d5eaf02f5a0",
    "comparison": "597cc0dc98774d5b8fa79795c8b195abd8143ab0973c1a95b7b9bc38423cc7b9",
    "sensitivity": "9de2d5a46b3870dec10250ee7dca1277624eb11f14833f31b06aecf812d0dea4",
}


def accepted_scenarios(base: dict, historical_audit: dict) -> dict[str, dict]:
    scenarios = {}
    for name, historical in historical_audit["sensitivity_scenarios"].items():
        assumptions = deepcopy(base)
        for section, values in historical["assumption_values"].items():
            assumptions[section].update(values)
        scenarios[name] = assumptions
    return scenarios


def main() -> None:
    historical_paths = {
        "config": ROOT / "configs/business_simulation_v2.yaml",
        "simulator": ROOT / "autocollections/simulator/business_simulator_v2.py",
        "comparison": ROOT / "reports/policy_baselines_v2.json",
        "sensitivity": ROOT / "reports/simulator_sensitivity_v2.json",
    }
    actual_hashes = {name: sha256_file(path) for name, path in historical_paths.items()}
    if actual_hashes != ACCEPTED_HASHES:
        raise ValueError("Accepted phase4_v2 inputs changed before the feasibility audit")

    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    assumptions = yaml.safe_load(historical_paths["config"].read_text())
    historical_audit = json.loads(historical_paths["sensitivity"].read_text())
    frame, metadata = load_dataset()
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    if manifest["split_sha256"] != benchmark["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")

    risk_source = ROOT / "reports/baselines.json"
    risk_report = json.loads(risk_source.read_text())
    if risk_report["selected_model"] != "logistic_derived":
        raise ValueError("Phase 3 source artifact does not select logistic_derived")
    if risk_report["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Risk source artifact and feasibility audit use different splits")

    _, probabilities = fit_canonical_risk_model(frame, manifest, seed=benchmark["seed"])
    validation = frame.iloc[manifest["splits"]["validation"]]
    features = approved_feature_view(validation)
    target = validation[TARGET]
    max_human_rate = benchmark["policy_constraints"]["max_human_escalation_rate"]
    risk_actions, _ = choose_risk_based_actions(probabilities, assumptions["policy_thresholds"])
    policies = {
        "always_no_contact": np.full(len(validation), NO_CONTACT),
        "always_digital_reminder": np.full(len(validation), DIGITAL_REMINDER),
        "always_payment_plan_review": np.full(len(validation), PAYMENT_PLAN_REVIEW),
        "always_human_escalation": np.full(len(validation), HUMAN_ESCALATION),
        "risk_based_four_action": risk_actions,
        "top35_human_else_no_contact": top_risk_human_policy(
            probabilities,
            human_capacity=max_human_rate,
            fallback_action=NO_CONTACT,
        ),
        "top35_human_else_digital": top_risk_human_policy(
            probabilities,
            human_capacity=max_human_rate,
            fallback_action=DIGITAL_REMINDER,
        ),
        "top35_human_else_payment_plan": top_risk_human_policy(
            probabilities,
            human_capacity=max_human_rate,
            fallback_action=PAYMENT_PLAN_REVIEW,
        ),
    }

    scenario_results = {}
    for scenario_name, scenario_assumptions in accepted_scenarios(
        assumptions, historical_audit
    ).items():
        results = {
            name: simulate_policy_v2(
                features,
                target,
                actions,
                scenario_assumptions,
                max_human_escalation_rate=max_human_rate,
            )
            for name, actions in policies.items()
        }
        annotated, raw_ranking, feasible_ranking = rank_with_human_capacity(
            results, max_human_rate=max_human_rate
        )
        _, oracle = protected_oracle_v2(
            features,
            target,
            scenario_assumptions,
            max_human_escalation_rate=max_human_rate,
        )
        best_capacity = max(
            CAPACITY_POLICIES,
            key=lambda name: annotated[name]["net_business_utility"],
        )
        scenario_results[scenario_name] = {
            "policies": annotated,
            "raw_utility_ranking": raw_ranking,
            "feasible_policy_ranking": feasible_ranking,
            "highest_raw_utility_policy": raw_ranking[0],
            "highest_feasible_policy": feasible_ranking[0],
            "risk_based_policy_utility": annotated["risk_based_four_action"][
                "net_business_utility"
            ],
            "best_capacity_saturating_policy": best_capacity,
            "best_capacity_saturating_policy_utility": annotated[best_capacity][
                "net_business_utility"
            ],
            "oracle_utility": oracle["net_business_utility"],
        }

    diagnostic = capacity_saturation_diagnostic(scenario_results)
    base = scenario_results["base"]
    _, oracle = protected_oracle_v2(
        features,
        target,
        assumptions,
        max_human_escalation_rate=max_human_rate,
    )
    payment_plan_oracle_dominated = (
        oracle["action_distribution"]["PAYMENT_PLAN_REVIEW"]["count"] == 0
    )
    broad_threshold = int(np.ceil(0.75 * len(scenario_results)))
    no_feasible_trivial_dominance = (
        not diagnostic["CAPACITY_SATURATION_WARNING"]
        and diagnostic["feasible_static_beats_risk_count"] < broad_threshold
    )
    freeze_criteria = {
        "accepted_integrity_properties_retained": True,
        "hard_human_capacity_enforced": all(
            result["FEASIBLE"] == (result["human_escalation_rate"] <= max_human_rate + 1e-12)
            for scenario in scenario_results.values()
            for result in scenario["policies"].values()
        ),
        "oracle_uses_multiple_actions": sum(
            value["rate"] >= 0.01 for value in oracle["action_distribution"].values()
        )
        >= 2,
        "no_feasible_trivial_or_capacity_dominance": no_feasible_trivial_dominance,
    }
    report = {
        "artifact_version": 1,
        "audit": "phase4_7_feasibility_and_capacity",
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
            "source_artifact_sha256": sha256_file(risk_source),
        },
        "accepted_phase4_v2_hashes": actual_hashes,
        "human_capacity": max_human_rate,
        "raw_utility_ranking": base["raw_utility_ranking"],
        "feasible_policy_ranking": base["feasible_policy_ranking"],
        "base_policy_results": base["policies"],
        "protected_oracle": oracle,
        "sensitivity_scenarios": {
            name: {key: value for key, value in result.items() if key != "policies"}
            for name, result in scenario_results.items()
        },
        "capacity_diagnostic": diagnostic,
        "PAYMENT_PLAN_ORACLE_DOMINATED": payment_plan_oracle_dominated,
        "freeze_criteria": {
            **freeze_criteria,
            "FREEZE_V2": all(freeze_criteria.values()),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("Phase 4.7 feasibility and capacity audit")
    print("Evaluation: validation only; hidden test NOT EVALUATED")
    print("policy\tutility\thuman_rate\tFEASIBLE")
    for name in base["raw_utility_ranking"]:
        result = base["policies"][name]
        print(
            f"{name}\t{result['net_business_utility']:.2f}\t"
            f"{result['human_escalation_rate']:.4f}\t"
            f"{str(result['FEASIBLE']).lower()}"
        )
    print("raw_ranking\t" + " > ".join(base["raw_utility_ranking"]))
    print("feasible_ranking\t" + " > ".join(base["feasible_policy_ranking"]))
    print("scenario\traw_winner\tfeasible_winner\tbest_capacity\trisk\tcapacity\toracle")
    for name, result in scenario_results.items():
        print(
            f"{name}\t{result['highest_raw_utility_policy']}\t"
            f"{result['highest_feasible_policy']}\t"
            f"{result['best_capacity_saturating_policy']}\t"
            f"{result['risk_based_policy_utility']:.2f}\t"
            f"{result['best_capacity_saturating_policy_utility']:.2f}\t"
            f"{result['oracle_utility']:.2f}"
        )
    print("CAPACITY_SATURATION_WARNING = " + str(diagnostic["CAPACITY_SATURATION_WARNING"]).lower())
    print("PAYMENT_PLAN_ORACLE_DOMINATED = " + str(payment_plan_oracle_dominated).lower())
    print("FREEZE_V2 = " + str(all(freeze_criteria.values())).lower())
    print(f"Artifact SHA-256: {sha256_file(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
