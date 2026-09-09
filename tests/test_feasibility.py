import inspect

import numpy as np

from autocollections.evaluation.feasibility import (
    CAPACITY_POLICIES,
    capacity_saturation_diagnostic,
    rank_with_human_capacity,
    top_risk_human_policy,
)
from autocollections.simulator.business_simulator import HUMAN_ESCALATION, NO_CONTACT


def test_top35_policy_uses_only_risk_and_exact_capacity() -> None:
    probabilities = np.linspace(0, 1, 20)
    actions = top_risk_human_policy(probabilities, human_capacity=0.35, fallback_action=NO_CONTACT)

    assert (actions == HUMAN_ESCALATION).sum() == 7
    assert np.flatnonzero(actions == HUMAN_ESCALATION).tolist() == list(range(13, 20))
    assert "target" not in inspect.signature(top_risk_human_policy).parameters


def test_human_capacity_overrides_raw_utility_ranking() -> None:
    results = {
        "always_human_escalation": {
            "net_business_utility": 100.0,
            "human_escalation_rate": 1.0,
        },
        "risk_based_four_action": {
            "net_business_utility": 50.0,
            "human_escalation_rate": 0.1,
        },
        "top35_human_else_no_contact": {
            "net_business_utility": 75.0,
            "human_escalation_rate": 0.35,
        },
    }
    annotated, raw, feasible = rank_with_human_capacity(results, max_human_rate=0.35)

    assert raw[0] == "always_human_escalation"
    assert feasible[0] == "top35_human_else_no_contact"
    assert annotated["always_human_escalation"]["FEASIBLE"] is False
    assert annotated["always_human_escalation"]["PRIMARY_SCORE_ELIGIBLE"] is False
    assert annotated["always_human_escalation"]["KEEP_ELIGIBLE"] is False
    assert annotated["top35_human_else_no_contact"]["FEASIBLE"] is True


def test_capacity_saturation_warning_is_diagnostic_only() -> None:
    policies = {
        "always_no_contact": {"net_business_utility": 0.0},
        "always_digital_reminder": {"net_business_utility": 10.0},
        "always_payment_plan_review": {"net_business_utility": 20.0},
        "risk_based_four_action": {"net_business_utility": 30.0},
        CAPACITY_POLICIES[0]: {"net_business_utility": 40.0},
        CAPACITY_POLICIES[1]: {"net_business_utility": 50.0},
        CAPACITY_POLICIES[2]: {"net_business_utility": 45.0},
    }
    diagnostic = capacity_saturation_diagnostic(
        {
            "base": {"policies": policies},
            "low": {"policies": policies},
            "high": {"policies": policies},
        }
    )

    assert diagnostic["CAPACITY_SATURATION_WARNING"] is True
    assert diagnostic["capacity_saturating_beats_risk_count"] == 3
    assert diagnostic["feasible_static_beats_risk_count"] == 0
