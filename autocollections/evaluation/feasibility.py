from collections import Counter
from typing import Any

import numpy as np

from autocollections.simulator.business_simulator import (
    ACTION_NAMES,
    HUMAN_ESCALATION,
)

CAPACITY_POLICIES = (
    "top35_human_else_no_contact",
    "top35_human_else_digital",
    "top35_human_else_payment_plan",
)


def top_risk_human_policy(
    probabilities: np.ndarray, *, human_capacity: float, fallback_action: int
) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1]")
    if not 0 <= human_capacity <= 1:
        raise ValueError("Human capacity must be within [0, 1]")
    if fallback_action not in ACTION_NAMES or fallback_action == HUMAN_ESCALATION:
        raise ValueError("Fallback must be a non-human action")

    actions = np.full(len(probability), fallback_action, dtype=int)
    top_count = int(np.floor(human_capacity * len(probability)))
    top_indices = np.argsort(-probability, kind="stable")[:top_count]
    actions[top_indices] = HUMAN_ESCALATION
    return actions


def rank_with_human_capacity(
    policy_results: dict[str, dict[str, Any]], *, max_human_rate: float
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    annotated = {}
    for name, result in policy_results.items():
        feasible = result["human_escalation_rate"] <= max_human_rate + 1e-12
        annotated[name] = {
            **result,
            "FEASIBLE": feasible,
            "PRIMARY_SCORE_ELIGIBLE": feasible,
            "KEEP_ELIGIBLE": feasible,
            "feasibility_violations": [] if feasible else ["human_escalation_capacity"],
        }
    raw_ranking = sorted(
        annotated,
        key=lambda name: annotated[name]["net_business_utility"],
        reverse=True,
    )
    feasible_ranking = [name for name in raw_ranking if annotated[name]["FEASIBLE"]]
    return annotated, raw_ranking, feasible_ranking


def capacity_saturation_diagnostic(
    scenarios: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    risk_policy = "risk_based_four_action"
    feasible_static = (
        "always_no_contact",
        "always_digital_reminder",
        "always_payment_plan_review",
    )
    capacity_winners = {}
    capacity_beats_risk = 0
    static_beats_risk = 0
    for scenario_name, scenario in scenarios.items():
        policies = scenario["policies"]
        best_capacity = max(
            CAPACITY_POLICIES,
            key=lambda name: policies[name]["net_business_utility"],
        )
        capacity_winners[scenario_name] = best_capacity
        capacity_beats_risk += (
            policies[best_capacity]["net_business_utility"]
            > policies[risk_policy]["net_business_utility"]
        )
        static_beats_risk += (
            max(policies[name]["net_business_utility"] for name in feasible_static)
            > policies[risk_policy]["net_business_utility"]
        )

    scenario_count = len(scenarios)
    broad_threshold = int(np.ceil(0.75 * scenario_count))
    common_capacity, common_count = Counter(capacity_winners.values()).most_common(1)[0]
    warning = capacity_beats_risk >= broad_threshold
    return {
        "CAPACITY_SATURATION_WARNING": warning,
        "scenario_count": scenario_count,
        "capacity_saturating_beats_risk_count": capacity_beats_risk,
        "feasible_static_beats_risk_count": static_beats_risk,
        "most_common_capacity_saturating_winner": common_capacity,
        "most_common_capacity_saturating_winner_count": common_count,
        "scenario_capacity_winners": capacity_winners,
    }
