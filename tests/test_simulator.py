import numpy as np
import pandas as pd
import pytest

from autocollections.data.schema import ALLOWED_MODEL_FEATURES
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
    dominance_diagnostic,
    simulate_policy,
)

ASSUMPTIONS = {
    "simulation_version": "test_v1",
    "costs": {
        "NO_CONTACT": 0.0,
        "DIGITAL_REMINDER": 5.0,
        "PAYMENT_PLAN_REVIEW": 50.0,
        "HUMAN_ESCALATION": 150.0,
    },
    "experience_penalties": {
        "unnecessary_digital_contact": 5.0,
        "unnecessary_payment_plan_review": 25.0,
        "unnecessary_human_escalation": 100.0,
    },
    "missed_opportunity": {"no_contact_default_rate": 0.015},
    "recovery_rate_bounds": {
        "DIGITAL_REMINDER": [0.02, 0.10],
        "PAYMENT_PLAN_REVIEW": [0.06, 0.20],
        "HUMAN_ESCALATION": [0.08, 0.25],
    },
    "exposure": {"max_fraction_of_limit": 1.0, "max_exposure_inr": 250_000.0},
    "policy_penalties": {"excess_human_escalation": 250.0},
}


def simulator_features(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=range(rows), columns=ALLOWED_MODEL_FEATURES)
    frame["LIMIT_BAL"] = 100.0
    for month in range(1, 7):
        frame[f"BILL_AMT{month}"] = 100.0
    return frame


def test_simulator_matches_hand_calculation_and_is_deterministic() -> None:
    features = simulator_features(4)
    target = np.array([1, 0, 1, 0])
    actions = np.array([NO_CONTACT, DIGITAL_REMINDER, PAYMENT_PLAN_REVIEW, HUMAN_ESCALATION])
    probabilities = np.array([0.1, 0.3, 0.6, 0.8])

    first = simulate_policy(
        features,
        target,
        actions,
        probabilities,
        ASSUMPTIONS,
        max_human_escalation_rate=1.0,
    )
    second = simulate_policy(
        features,
        target,
        actions,
        probabilities,
        ASSUMPTIONS,
        max_human_escalation_rate=1.0,
    )

    assert first == second
    assert first["simulated_recovery_benefit_total"] == 13.0
    assert first["treatment_cost_total"] == 205.0
    assert first["customer_experience_penalty_total"] == 5.0
    assert first["unnecessary_escalation_penalty_total"] == 100.0
    assert first["missed_opportunity_penalty_total"] == 1.5
    assert first["net_business_utility"] == -298.5
    assert first["average_utility_per_account"] == -74.625


def test_simulator_ignores_demographics_and_counts_capacity_violations() -> None:
    features = simulator_features(10)
    target = np.zeros(10, dtype=int)
    actions = np.full(10, HUMAN_ESCALATION)
    probabilities = np.full(10, 0.8)
    expected = simulate_policy(
        features,
        target,
        actions,
        probabilities,
        ASSUMPTIONS,
        max_human_escalation_rate=0.35,
    )
    with_demographics = features.assign(SEX=99, EDUCATION=99, MARRIAGE=99, AGE=99)
    actual = simulate_policy(
        with_demographics,
        target,
        actions,
        probabilities,
        ASSUMPTIONS,
        max_human_escalation_rate=0.35,
    )

    assert actual == expected
    assert actual["policy_violation_count"] == 7
    assert actual["policy_violation_penalty_total"] == 1_750.0


def test_non_defaults_never_recover_and_treatment_cost_increases_with_intensity() -> None:
    costs = []
    for action in (NO_CONTACT, DIGITAL_REMINDER, PAYMENT_PLAN_REVIEW, HUMAN_ESCALATION):
        result = simulate_policy(
            simulator_features(1),
            np.array([0]),
            np.array([action]),
            np.array([0.8]),
            ASSUMPTIONS,
            max_human_escalation_rate=1.0,
        )
        assert result["simulated_recovery_benefit_total"] == 0.0
        assert result["net_business_utility"] <= 0.0
        costs.append(result["treatment_cost_total"])

    assert costs == [0.0, 5.0, 50.0, 150.0]


def test_dominance_diagnostic_warns_without_changing_results() -> None:
    policy_results = {
        "always_no_contact": {"net_business_utility": 0.0},
        "always_digital_reminder": {"net_business_utility": 1.0},
        "always_payment_plan_review": {
            "net_business_utility": 3.0,
            "action_distribution": {"PAYMENT_PLAN_REVIEW": {"rate": 1.0}},
        },
        "always_human_escalation": {"net_business_utility": 2.0},
        "risk_based_four_action": {"net_business_utility": 1.5},
    }
    diagnostic = dominance_diagnostic(
        {
            "base": {"policies": policy_results},
            "high_cost": {"policies": policy_results},
            "low_recovery": {"policies": policy_results},
        }
    )

    assert diagnostic["warning_only"] is True
    assert diagnostic["static_beats_risk_scenario_count"] == 3
    assert len(diagnostic["warnings"]) == 3


def test_simulator_rejects_actions_outside_the_fixed_space() -> None:
    with pytest.raises(ValueError, match="Actions must be one of"):
        simulate_policy(
            simulator_features(1),
            np.array([0]),
            np.array([4]),
            np.array([0.5]),
            ASSUMPTIONS,
            max_human_escalation_rate=0.35,
        )
