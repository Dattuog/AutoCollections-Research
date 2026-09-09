from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from autocollections.features.approved_features import approved_feature_view

NO_CONTACT = 0
DIGITAL_REMINDER = 1
PAYMENT_PLAN_REVIEW = 2
HUMAN_ESCALATION = 3
ACTION_NAMES = {
    NO_CONTACT: "NO_CONTACT",
    DIGITAL_REMINDER: "DIGITAL_REMINDER",
    PAYMENT_PLAN_REVIEW: "PAYMENT_PLAN_REVIEW",
    HUMAN_ESCALATION: "HUMAN_ESCALATION",
}


def _aggregate(
    mask: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
    observed_target: np.ndarray,
    recovery: np.ndarray,
    treatment_cost: np.ndarray,
    experience_penalty: np.ndarray,
    escalation_penalty: np.ndarray,
    missed_opportunity_penalty: np.ndarray,
    policy_penalty: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    utility = (
        recovery
        - treatment_cost
        - experience_penalty
        - escalation_penalty
        - missed_opportunity_penalty
        - policy_penalty
    )
    action_counts = {
        ACTION_NAMES[action]: int(((actions == action) & mask).sum()) for action in ACTION_NAMES
    }
    return {
        "account_count": count,
        "simulated_recovery_benefit_total": float(recovery[mask].sum()),
        "treatment_cost_total": float(treatment_cost[mask].sum()),
        "customer_experience_penalty_total": float(experience_penalty[mask].sum()),
        "unnecessary_escalation_penalty_total": float(escalation_penalty[mask].sum()),
        "missed_opportunity_penalty_total": float(missed_opportunity_penalty[mask].sum()),
        "policy_violation_penalty_total": float(policy_penalty[mask].sum()),
        "net_business_utility": float(utility[mask].sum()),
        "average_utility_per_account": float(utility[mask].mean()) if count else 0.0,
        "action_distribution": {
            name: {
                "count": action_counts[name],
                "rate": action_counts[name] / count if count else 0.0,
            }
            for name in action_counts
        },
        "human_escalation_rate": action_counts["HUMAN_ESCALATION"] / count if count else 0.0,
        "digital_contact_rate": action_counts["DIGITAL_REMINDER"] / count if count else 0.0,
        "policy_violation_count": int(((policy_penalty > 0) & mask).sum()),
        "predicted_risk_mean": float(probabilities[mask].mean()) if count else 0.0,
        "observed_default_rate": float(observed_target[mask].mean()) if count else 0.0,
    }


def simulate_policy(
    features: pd.DataFrame,
    observed_target: pd.Series | np.ndarray,
    actions: np.ndarray,
    predicted_probabilities: np.ndarray,
    assumptions: dict[str, Any],
    *,
    max_human_escalation_rate: float,
    risk_bands: np.ndarray | None = None,
) -> dict[str, Any]:
    financial = approved_feature_view(features).reset_index(drop=True)
    target = np.asarray(observed_target, dtype=int)
    actions = np.asarray(actions, dtype=int)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    row_count = len(financial)
    if len(target) != row_count or len(actions) != row_count or len(probabilities) != row_count:
        raise ValueError("Features, outcomes, actions, and probabilities must have equal rows")
    if not set(np.unique(actions)).issubset(ACTION_NAMES):
        raise ValueError("Actions must be one of 0, 1, 2, or 3")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Observed target must be binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1]")

    limit = financial["LIMIT_BAL"].clip(lower=0).to_numpy(dtype=float)
    bill = financial["BILL_AMT1"].clip(lower=0).to_numpy(dtype=float)
    exposure_cap = np.minimum(
        limit * assumptions["exposure"]["max_fraction_of_limit"],
        assumptions["exposure"]["max_exposure_inr"],
    )
    exposure = np.clip(bill, 0, exposure_cap)
    exposure_fraction = np.divide(
        exposure, exposure_cap, out=np.zeros(row_count), where=exposure_cap > 0
    )

    bills = financial[[f"BILL_AMT{month}" for month in range(1, 7)]].clip(lower=1)
    payments = financial[[f"PAY_AMT{month}" for month in range(1, 7)]]
    payment_ratios = np.clip(payments.to_numpy() / bills.to_numpy(), 0, 1)
    utilization = np.clip(
        financial[[f"BILL_AMT{month}" for month in range(1, 7)]].clip(lower=0).to_numpy()
        / np.maximum(limit[:, None], 1),
        0,
        1,
    )
    months_with_payment = (payments.to_numpy() > 0).mean(axis=1)
    ability_proxy = np.clip(
        0.5 * payment_ratios.mean(axis=1)
        + 0.25 * months_with_payment
        + 0.25 * (1 - utilization.mean(axis=1)),
        0,
        1,
    )
    improving_delay = (
        financial["PAY_6"].clip(lower=0).to_numpy() > financial["PAY_0"].clip(lower=0).to_numpy()
    ).astype(float)
    engagement_proxy = np.clip(
        0.5 * months_with_payment + 0.3 * payment_ratios[:, 0] + 0.2 * improving_delay,
        0,
        1,
    )

    recovery_rates = np.zeros(row_count)
    bounds = assumptions["recovery_rate_bounds"]
    profiles = {
        DIGITAL_REMINDER: engagement_proxy,
        PAYMENT_PLAN_REVIEW: 0.5 * engagement_proxy + 0.5 * (1 - ability_proxy),
        HUMAN_ESCALATION: 0.5 * (1 - ability_proxy) + 0.5 * exposure_fraction,
    }
    for action, profile in profiles.items():
        low, high = bounds[ACTION_NAMES[action]]
        mask = actions == action
        recovery_rates[mask] = low + (high - low) * profile[mask]
    recovery = exposure * recovery_rates * (target == 1)

    costs = assumptions["costs"]
    treatment_cost = np.array([costs[ACTION_NAMES[action]] for action in actions], dtype=float)
    non_default = target == 0
    experience_penalty = np.zeros(row_count)
    experience_penalty[(actions == DIGITAL_REMINDER) & non_default] = assumptions[
        "experience_penalties"
    ]["unnecessary_digital_contact"]
    experience_penalty[(actions == PAYMENT_PLAN_REVIEW) & non_default] = assumptions[
        "experience_penalties"
    ]["unnecessary_payment_plan_review"]
    escalation_penalty = np.zeros(row_count)
    escalation_penalty[(actions == HUMAN_ESCALATION) & non_default] = assumptions[
        "experience_penalties"
    ]["unnecessary_human_escalation"]
    missed_opportunity_penalty = np.zeros(row_count)
    missed_opportunity_penalty[(actions == NO_CONTACT) & (target == 1)] = (
        exposure[(actions == NO_CONTACT) & (target == 1)]
        * assumptions["missed_opportunity"]["no_contact_default_rate"]
    )

    human_indices = np.flatnonzero(actions == HUMAN_ESCALATION)
    allowed_human_count = int(np.floor(max_human_escalation_rate * row_count))
    violating_indices = human_indices[allowed_human_count:]
    policy_penalty = np.zeros(row_count)
    policy_penalty[violating_indices] = assumptions["policy_penalties"]["excess_human_escalation"]

    all_rows = np.ones(row_count, dtype=bool)
    result = _aggregate(
        all_rows,
        actions,
        probabilities,
        target,
        recovery,
        treatment_cost,
        experience_penalty,
        escalation_penalty,
        missed_opportunity_penalty,
        policy_penalty,
    )
    if risk_bands is not None:
        bands = np.asarray(risk_bands)
        if len(bands) != row_count:
            raise ValueError("Risk-band row count differs from evaluation rows")
        result["by_predicted_risk_band"] = {
            str(band): _aggregate(
                bands == band,
                actions,
                probabilities,
                target,
                recovery,
                treatment_cost,
                experience_penalty,
                escalation_penalty,
                missed_opportunity_penalty,
                policy_penalty,
            )
            for band in ("lowest", "low_moderate", "higher", "highest")
        }
    return result


def dominance_diagnostic(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "base" not in scenarios:
        raise ValueError("Sensitivity scenarios must include base")

    risk_policy = "risk_based_four_action"
    static_policies = (
        "always_no_contact",
        "always_digital_reminder",
        "always_payment_plan_review",
        "always_human_escalation",
    )
    winners = {
        name: max(
            scenario["policies"],
            key=lambda policy: scenario["policies"][policy]["net_business_utility"],
        )
        for name, scenario in scenarios.items()
    }
    static_beats_risk = sum(
        max(scenario["policies"][policy]["net_business_utility"] for policy in static_policies)
        > scenario["policies"][risk_policy]["net_business_utility"]
        for scenario in scenarios.values()
    )
    most_common_static, most_common_count = Counter(winners.values()).most_common(1)[0]
    base_winner = winners["base"]
    base_action_rate = max(
        item["rate"]
        for item in scenarios["base"]["policies"][base_winner]["action_distribution"].values()
    )
    broad_threshold = int(np.ceil(0.75 * len(scenarios)))
    warnings = []
    if base_winner in static_policies and base_action_rate >= 0.95:
        warnings.append(
            f"{base_winner} treats nearly the entire population and has the highest "
            "unconstrained base-scenario utility."
        )
    if static_beats_risk >= broad_threshold:
        warnings.append(
            f"A static policy beats {risk_policy} in {static_beats_risk}/{len(scenarios)} "
            "sensitivity scenarios."
        )
    if most_common_static in static_policies and most_common_count >= broad_threshold:
        warnings.append(
            f"{most_common_static} is the unconstrained winner in "
            f"{most_common_count}/{len(scenarios)} scenarios."
        )
    return {
        "warning_only": True,
        "base_unconstrained_winner": base_winner,
        "base_winner_max_action_rate": base_action_rate,
        "static_beats_risk_scenario_count": static_beats_risk,
        "scenario_count": len(scenarios),
        "most_common_unconstrained_winner": most_common_static,
        "most_common_winner_count": most_common_count,
        "warnings": warnings,
    }
