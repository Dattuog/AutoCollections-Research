from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autocollections.data.schema import ALLOWED_MODEL_FEATURES
from autocollections.simulator.business_simulator import (
    DIGITAL_REMINDER,
    HUMAN_ESCALATION,
    NO_CONTACT,
    PAYMENT_PLAN_REVIEW,
)
from autocollections.simulator.business_simulator_v2 import (
    action_suitability_v2,
    behavioral_proxies_v2,
    counterfactual_action_utilities_v2,
    protected_oracle_v2,
    simulate_policy_v2,
)

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS = yaml.safe_load((ROOT / "configs/business_simulation_v2.yaml").read_text())


def heterogeneous_customers() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(0.0, index=range(4), columns=ALLOWED_MODEL_FEATURES)
    frame["LIMIT_BAL"] = [1_000, 1_000, 10_000, 1_000]
    for month in range(1, 7):
        frame[f"BILL_AMT{month}"] = [200, 1_000, 10_000, 200]
        frame[f"PAY_AMT{month}"] = [200, 0, 10_000, 200]
    for delay in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        frame[delay] = [0, 8, 0, 0]
    frame.loc[2, "PAY_6"] = 1
    return frame, np.array([1, 1, 1, 0])


def test_v2_suitability_is_heterogeneous_and_demographic_invariant() -> None:
    features, _ = heterogeneous_customers()
    proxies = behavioral_proxies_v2(features, ASSUMPTIONS)
    suitability = action_suitability_v2(features, ASSUMPTIONS)
    with_demographics = features.assign(SEX=99, EDUCATION=99, MARRIAGE=99, AGE=99)

    assert (
        ((proxies >= 0) & (proxies <= 1))
        .loc[
            :,
            [
                "ability_to_pay",
                "engagement",
                "severity",
            ],
        ]
        .all()
        .all()
    )
    assert ((suitability >= 0) & (suitability <= 1)).all().all()
    assert suitability.iloc[0].nunique() > 1
    pd.testing.assert_frame_equal(
        suitability, action_suitability_v2(with_demographics, ASSUMPTIONS)
    )


def test_v2_expensive_actions_are_not_universally_best() -> None:
    features, target = heterogeneous_customers()
    utilities = counterfactual_action_utilities_v2(features, target, ASSUMPTIONS)

    assert utilities.iloc[0].idxmax() == "DIGITAL_REMINDER"
    assert utilities.iloc[1].idxmax() == "HUMAN_ESCALATION"
    assert utilities.iloc[2].idxmax() == "PAYMENT_PLAN_REVIEW"
    assert utilities.iloc[3].idxmax() == "NO_CONTACT"
    assert utilities.loc[0, "DIGITAL_REMINDER"] > utilities.loc[0, "HUMAN_ESCALATION"]
    np.testing.assert_allclose(utilities.iloc[1].to_numpy(), [-15.0, -5.0, -38.8, 70.0])


def test_v2_oracle_is_mixed_and_simulator_is_deterministic() -> None:
    features, target = heterogeneous_customers()
    oracle_actions, oracle = protected_oracle_v2(
        features,
        target,
        ASSUMPTIONS,
        max_human_escalation_rate=1.0,
    )
    repeated = simulate_policy_v2(
        features,
        target,
        oracle_actions,
        ASSUMPTIONS,
        max_human_escalation_rate=1.0,
    )

    assert oracle_actions.tolist() == [
        DIGITAL_REMINDER,
        HUMAN_ESCALATION,
        PAYMENT_PLAN_REVIEW,
        NO_CONTACT,
    ]
    assert oracle["net_business_utility"] == repeated["net_business_utility"]
    assert oracle["deployable_policy"] is False
    assert oracle["uses_realized_target"] is True
    constrained = simulate_policy_v2(
        features,
        target,
        np.full(len(features), HUMAN_ESCALATION),
        ASSUMPTIONS,
        max_human_escalation_rate=0.25,
    )
    assert constrained["policy_violation_count"] == 3


def test_v2_non_defaults_never_receive_recovery() -> None:
    features, _ = heterogeneous_customers()
    target = np.zeros(len(features), dtype=int)
    for action in (NO_CONTACT, DIGITAL_REMINDER, PAYMENT_PLAN_REVIEW, HUMAN_ESCALATION):
        result = simulate_policy_v2(
            features,
            target,
            np.full(len(features), action),
            ASSUMPTIONS,
            max_human_escalation_rate=1.0,
        )
        assert result["simulated_recovery_benefit_total"] == 0.0
        assert result["net_business_utility"] <= 0.0
