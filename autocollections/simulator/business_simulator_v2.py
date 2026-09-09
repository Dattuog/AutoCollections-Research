from typing import Any

import numpy as np
import pandas as pd

from autocollections.features.approved_features import approved_feature_view
from autocollections.simulator.business_simulator import (
    ACTION_NAMES,
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
)

PROXY_DEFINITIONS = {
    "ability_to_pay": (
        "0.50 mean capped payment/bill ratio + 0.25 payment-month frequency + "
        "0.25 inverse mean utilization"
    ),
    "engagement": (
        "0.40 payment-month frequency + 0.35 recent capped payment/bill ratio + "
        "0.25 improving delinquency trend"
    ),
    "severity": (
        "0.45 recent delinquency + 0.30 persistent delinquency + "
        "0.15 mean utilization + 0.10 exposure fraction"
    ),
}
SUITABILITY_DEFINITIONS = {
    "DIGITAL_REMINDER": "(0.45 ability + 0.55 engagement) * (1 - severity)^2",
    "PAYMENT_PLAN_REVIEW": (
        "(0.40 severity + 0.30 engagement + 0.30 exposure_fraction) * (0.10 + 0.90 ability)"
    ),
    "HUMAN_ESCALATION": (
        "(0.60 severity + 0.40 exposure_fraction) * (1 - 0.60 * ability * engagement)"
    ),
}


def behavioral_proxies_v2(features: pd.DataFrame, assumptions: dict[str, Any]) -> pd.DataFrame:
    financial = approved_feature_view(features).reset_index(drop=True)
    limit = financial["LIMIT_BAL"].clip(lower=0).to_numpy(dtype=float)
    bills = financial[[f"BILL_AMT{month}" for month in range(1, 7)]].clip(lower=0)
    payments = financial[[f"PAY_AMT{month}" for month in range(1, 7)]].clip(lower=0)
    delays = financial[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].clip(
        lower=0, upper=8
    )

    exposure_cap = np.minimum(
        limit * assumptions["exposure"]["max_fraction_of_limit"],
        assumptions["exposure"]["max_exposure_inr"],
    )
    exposure = np.clip(bills["BILL_AMT1"].to_numpy(), 0, exposure_cap)
    exposure_fraction = np.divide(
        exposure, exposure_cap, out=np.zeros(len(financial)), where=exposure_cap > 0
    )
    payment_ratios = np.clip(payments.to_numpy() / np.maximum(bills.to_numpy(), 1), 0, 1)
    utilization = np.clip(bills.to_numpy() / np.maximum(limit[:, None], 1), 0, 1)
    payment_frequency = (payments.to_numpy() > 0).mean(axis=1)
    recent_delay = delays["PAY_0"].to_numpy() / 8
    persistent_delay = (delays.to_numpy() > 0).mean(axis=1)
    improving_delay = np.clip(
        (delays["PAY_6"].to_numpy() - delays["PAY_0"].to_numpy()) / 8,
        0,
        1,
    )

    return pd.DataFrame(
        {
            "exposure": exposure,
            "exposure_fraction": exposure_fraction,
            "ability_to_pay": np.clip(
                0.50 * payment_ratios.mean(axis=1)
                + 0.25 * payment_frequency
                + 0.25 * (1 - utilization.mean(axis=1)),
                0,
                1,
            ),
            "engagement": np.clip(
                0.40 * payment_frequency + 0.35 * payment_ratios[:, 0] + 0.25 * improving_delay,
                0,
                1,
            ),
            "severity": np.clip(
                0.45 * recent_delay
                + 0.30 * persistent_delay
                + 0.15 * utilization.mean(axis=1)
                + 0.10 * exposure_fraction,
                0,
                1,
            ),
        }
    )


def _suitability_from_proxies(proxies: pd.DataFrame) -> pd.DataFrame:
    ability = proxies["ability_to_pay"].to_numpy()
    engagement = proxies["engagement"].to_numpy()
    severity = proxies["severity"].to_numpy()
    exposure_fraction = proxies["exposure_fraction"].to_numpy()
    return pd.DataFrame(
        {
            "NO_CONTACT": np.zeros(len(proxies)),
            "DIGITAL_REMINDER": np.clip(
                (0.45 * ability + 0.55 * engagement) * (1 - severity) ** 2,
                0,
                1,
            ),
            "PAYMENT_PLAN_REVIEW": np.clip(
                (0.40 * severity + 0.30 * engagement + 0.30 * exposure_fraction)
                * (0.10 + 0.90 * ability),
                0,
                1,
            ),
            "HUMAN_ESCALATION": np.clip(
                (0.60 * severity + 0.40 * exposure_fraction) * (1 - 0.60 * ability * engagement),
                0,
                1,
            ),
        }
    )


def action_suitability_v2(features: pd.DataFrame, assumptions: dict[str, Any]) -> pd.DataFrame:
    return _suitability_from_proxies(behavioral_proxies_v2(features, assumptions))


def _counterfactual_components(
    features: pd.DataFrame,
    observed_target: pd.Series | np.ndarray,
    assumptions: dict[str, Any],
) -> dict[str, np.ndarray]:
    proxies = behavioral_proxies_v2(features, assumptions)
    suitability = _suitability_from_proxies(proxies).to_numpy()
    target = np.asarray(observed_target, dtype=int)
    if len(target) != len(proxies) or not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Observed target must be binary and match feature rows")

    base_effect = np.array(
        [
            0.0,
            assumptions["action_base_effects"]["DIGITAL_REMINDER"],
            assumptions["action_base_effects"]["PAYMENT_PLAN_REVIEW"],
            assumptions["action_base_effects"]["HUMAN_ESCALATION"],
        ]
    )
    recovery = proxies["exposure"].to_numpy()[:, None] * base_effect * suitability * target[:, None]
    treatment_cost = np.broadcast_to(
        np.array([assumptions["costs"][ACTION_NAMES[action]] for action in ACTION_NAMES]),
        recovery.shape,
    )
    experience_penalty = np.zeros_like(recovery)
    non_default = target == 0
    experience_penalty[non_default, DIGITAL_REMINDER] = assumptions["experience_penalties"][
        "unnecessary_digital_contact"
    ]
    experience_penalty[non_default, PAYMENT_PLAN_REVIEW] = assumptions["experience_penalties"][
        "unnecessary_payment_plan_review"
    ]
    escalation_penalty = np.zeros_like(recovery)
    escalation_penalty[non_default, HUMAN_ESCALATION] = assumptions["experience_penalties"][
        "unnecessary_human_escalation"
    ]
    missed_opportunity_penalty = np.zeros_like(recovery)
    missed_opportunity_penalty[:, NO_CONTACT] = (
        proxies["exposure"].to_numpy()
        * suitability[:, 1:].max(axis=1)
        * assumptions["missed_opportunity"]["no_contact_recoverable_exposure_rate"]
        * target
    )
    return {
        "recovery": recovery,
        "treatment_cost": treatment_cost,
        "experience_penalty": experience_penalty,
        "escalation_penalty": escalation_penalty,
        "missed_opportunity_penalty": missed_opportunity_penalty,
    }


def counterfactual_action_utilities_v2(
    features: pd.DataFrame,
    observed_target: pd.Series | np.ndarray,
    assumptions: dict[str, Any],
) -> pd.DataFrame:
    components = _counterfactual_components(features, observed_target, assumptions)
    utility = (
        components["recovery"]
        - components["treatment_cost"]
        - components["experience_penalty"]
        - components["escalation_penalty"]
        - components["missed_opportunity_penalty"]
    )
    return pd.DataFrame(utility, columns=[ACTION_NAMES[action] for action in ACTION_NAMES])


def simulate_policy_v2(
    features: pd.DataFrame,
    observed_target: pd.Series | np.ndarray,
    actions: np.ndarray,
    assumptions: dict[str, Any],
    *,
    max_human_escalation_rate: float,
) -> dict[str, Any]:
    components = _counterfactual_components(features, observed_target, assumptions)
    actions = np.asarray(actions, dtype=int)
    row_count = len(features)
    if len(actions) != row_count or not set(np.unique(actions)).issubset(ACTION_NAMES):
        raise ValueError("Actions must match feature rows and be one of 0, 1, 2, or 3")
    rows = np.arange(row_count)
    selected = {name: values[rows, actions] for name, values in components.items()}

    human_indices = np.flatnonzero(actions == HUMAN_ESCALATION)
    allowed_human_count = int(np.floor(max_human_escalation_rate * row_count))
    policy_penalty = np.zeros(row_count)
    policy_penalty[human_indices[allowed_human_count:]] = assumptions["policy_penalties"][
        "excess_human_escalation"
    ]
    utility = (
        selected["recovery"]
        - selected["treatment_cost"]
        - selected["experience_penalty"]
        - selected["escalation_penalty"]
        - selected["missed_opportunity_penalty"]
        - policy_penalty
    )
    action_counts = {
        ACTION_NAMES[action]: int((actions == action).sum()) for action in ACTION_NAMES
    }
    return {
        "account_count": row_count,
        "simulated_recovery_benefit_total": float(selected["recovery"].sum()),
        "treatment_cost_total": float(selected["treatment_cost"].sum()),
        "customer_experience_penalty_total": float(selected["experience_penalty"].sum()),
        "unnecessary_escalation_penalty_total": float(selected["escalation_penalty"].sum()),
        "missed_opportunity_penalty_total": float(selected["missed_opportunity_penalty"].sum()),
        "policy_violation_penalty_total": float(policy_penalty.sum()),
        "net_business_utility": float(utility.sum()),
        "average_utility_per_account": float(utility.mean()),
        "action_distribution": {
            name: {"count": count, "rate": count / row_count}
            for name, count in action_counts.items()
        },
        "human_escalation_rate": action_counts["HUMAN_ESCALATION"] / row_count,
        "digital_contact_rate": action_counts["DIGITAL_REMINDER"] / row_count,
        "policy_violation_count": int((policy_penalty > 0).sum()),
    }


def protected_oracle_v2(
    features: pd.DataFrame,
    observed_target: pd.Series | np.ndarray,
    assumptions: dict[str, Any],
    *,
    max_human_escalation_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluation-only per-account oracle; never a deployable policy."""
    actions = (
        counterfactual_action_utilities_v2(features, observed_target, assumptions)
        .to_numpy()
        .argmax(axis=1)
    )
    result = simulate_policy_v2(
        features,
        observed_target,
        actions,
        assumptions,
        max_human_escalation_rate=max_human_escalation_rate,
    )
    result["deployable_policy"] = False
    result["uses_realized_target"] = True
    result["oracle_scope"] = "per-account unconstrained; capacity violations remain reportable"
    return actions, result
