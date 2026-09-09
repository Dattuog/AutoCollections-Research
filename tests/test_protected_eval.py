import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autocollections.data.schema import ALLOWED_MODEL_FEATURES, TARGET
from autocollections.evaluation import protected_eval

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_schema_and_actions() -> None:
    sample_ids = np.array([10, 11])
    valid = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "predicted_probability": [0.2, 0.8],
            "action": [1, 3],
        }
    )
    probabilities, actions = protected_eval.validate_candidate_output(valid, sample_ids)
    np.testing.assert_array_equal(probabilities, [0.2, 0.8])
    np.testing.assert_array_equal(actions, [1, 3])

    invalid = valid.copy()
    invalid.loc[0, "action"] = 4
    with pytest.raises(ValueError, match="one of 0, 1, 2, or 3"):
        protected_eval.validate_candidate_output(invalid, sample_ids)
    with pytest.raises(ValueError, match="columns must be exactly"):
        protected_eval.validate_candidate_output(valid.assign(target=[0, 1]), sample_ids)


def test_frozen_evaluator_identity_rejects_hash_mismatch(monkeypatch) -> None:
    identity = protected_eval.evaluator_identity()
    assert identity["evaluator_version"] == "phase5_evaluator_v1"
    assert identity["split_sha256"] == protected_eval.EXPECTED_SPLIT_SHA256
    assert identity["strong_baseline_id"] == "top35_human_else_digital"

    monkeypatch.setattr(
        protected_eval,
        "EXPECTED_PROTECTED_HASHES",
        {**protected_eval.EXPECTED_PROTECTED_HASHES, "simulator_config_sha256": "wrong"},
    )
    with pytest.raises(ValueError, match="Protected simulator or feasibility inputs changed"):
        protected_eval.evaluator_identity()


def test_strong_baseline_reference_matches_accepted_phase47_artifact() -> None:
    artifact = json.loads((ROOT / "reports/feasibility_audit_v2.json").read_text())
    baseline = artifact["base_policy_results"][protected_eval.STRONG_BASELINE_ID]

    assert baseline["FEASIBLE"] is True
    assert baseline["net_business_utility"] == protected_eval.STRONG_BASELINE_UTILITY_TOTAL
    assert (
        baseline["average_utility_per_account"]
        == protected_eval.STRONG_BASELINE_UTILITY_PER_ACCOUNT
    )


def test_research_boundary_capacity_and_hidden_isolation(tmp_path, monkeypatch) -> None:
    assumptions = yaml.safe_load((ROOT / "configs/business_simulation_v2.yaml").read_text())
    frame = pd.DataFrame(0.0, index=range(10), columns=ALLOWED_MODEL_FEATURES)
    frame["LIMIT_BAL"] = 1_000
    for month in range(1, 7):
        frame[f"BILL_AMT{month}"] = 500
        frame[f"PAY_AMT{month}"] = 100
    frame[TARGET] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    manifest = {
        "split_sha256": "fixture-split",
        "splits": {
            "train": [0, 1, 2, 3],
            "validation": [4, 5, 6, 7],
            "hidden_test": [8, 9],
        },
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "configs/benchmark.yaml").write_text(
        yaml.safe_dump({"policy_constraints": {"max_human_escalation_rate": 0.35}})
    )
    (tmp_path / "configs/business_simulation_v2.yaml").write_text(yaml.safe_dump(assumptions))
    (tmp_path / "data/processed/split_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(protected_eval, "ROOT", tmp_path)
    monkeypatch.setattr(
        protected_eval,
        "evaluator_identity",
        lambda: {
            "evaluator_version": "fixture-evaluator",
            "evaluator_sha256": "fixture-hash",
            "dataset_sha256": "fixture-data",
            "split_sha256": "fixture-split",
        },
    )
    monkeypatch.setattr(
        protected_eval,
        "load_dataset",
        lambda: (frame.copy(), {"data_sha256": "fixture-data"}),
    )

    X_train, y_train, X_validation, validation_ids = protected_eval.load_research_inputs()
    assert X_train.columns.tolist() == list(ALLOWED_MODEL_FEATURES)
    assert X_validation.columns.tolist() == list(ALLOWED_MODEL_FEATURES)
    assert len(y_train) == 4
    np.testing.assert_array_equal(validation_ids, [4, 5, 6, 7])

    candidate = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "predicted_probability": [0.1, 0.9, 0.2, 0.8],
            "action": [3, 3, 3, 3],
        }
    )
    first = protected_eval.evaluate_candidate(
        candidate, experiment_id="fixture", runtime_seconds=1.0
    )
    assert first["feasible"] is False
    assert first["primary_score"] == protected_eval.DISQUALIFIED_SCORE
    assert first["KEEP_ELIGIBLE"] is False

    frame.loc[manifest["splits"]["hidden_test"], TARGET] ^= 1
    frame.loc[manifest["splits"]["hidden_test"], ALLOWED_MODEL_FEATURES] += 9_000_000
    second = protected_eval.evaluate_candidate(
        candidate, experiment_id="fixture", runtime_seconds=1.0
    )
    assert first == second
    assert second["hidden_test_evaluated"] is False
    assert not any(key.startswith("hidden_") for key in second if key != "hidden_test_evaluated")


def test_train_policy_source_has_no_validation_target_or_hidden_test_access() -> None:
    source = (ROOT / "train.py").read_text()
    assert "default_next_month" not in source
    assert "y_validation" not in source
    assert "hidden_test" not in source
