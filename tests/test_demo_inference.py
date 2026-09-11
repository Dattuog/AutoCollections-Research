import inspect
from math import floor

import pandas as pd
import pytest

from autocollections.data.schema import ALLOWED_MODEL_FEATURES, EXCLUDED_MODEL_FEATURES
from demo import inference


def test_frozen_artifact_identity_and_reconstruction_equivalence() -> None:
    manifest = inference.verify_frozen_artifact()

    assert manifest["source_experiment"] == "exp_039"
    assert manifest["train_py_sha256"] == (
        "f11c6817911ba5068a46f6321493412a0af13531f5becd33828e18dfbe065ed3"
    )
    assert manifest["model_artifact_sha256"] == (
        "dbe9e60bc85cdb432771ee7fa88119d6a89820b36213c16e031b5bdb41b5e0c4"
    )
    assert manifest["hidden_test_evaluated"] is False
    equivalence = manifest["validation_equivalence"]
    assert equivalence["changed_treatment_decisions"] == 0
    assert equivalence["entire_serialized_candidate_exact_match"] is True
    assert equivalence["exact_sample_ids_and_order"] is True
    assert equivalence["maximum_absolute_probability_difference"] == 0.0
    assert equivalence["mean_absolute_probability_difference"] == 0.0
    assert equivalence["score_difference"] == 0.0


def test_frozen_artifact_hash_mismatch_fails_closed(monkeypatch) -> None:
    original_sha256 = inference._sha256
    monkeypatch.setattr(
        inference,
        "_sha256",
        lambda path: "tampered" if path == inference.MODEL_PATH else original_sha256(path),
    )

    with pytest.raises(ValueError, match="model artifact hash mismatch"):
        inference.verify_frozen_artifact()


def test_demo_is_synthetic_approved_feature_only_and_deterministic() -> None:
    approved = set(ALLOWED_MODEL_FEATURES)
    excluded = set(EXCLUDED_MODEL_FEATURES)
    assert all(set(profile) == approved for profile in inference.SCENARIOS.values())
    assert all(not excluded & set(profile) for profile in inference.SCENARIOS.values())

    selected = inference.SCENARIOS["Moderate-risk customer"]
    first = inference.synthetic_portfolio(selected)
    second = inference.synthetic_portfolio(selected)
    pd.testing.assert_frame_equal(first, second)

    first_scores = inference.score_portfolio(first)
    second_scores = inference.score_portfolio(second)
    pd.testing.assert_frame_equal(first_scores, second_scores)
    assert set(first_scores["action"]) <= {
        "NO_CONTACT",
        "DIGITAL_REMINDER",
        "HUMAN_ESCALATION",
    }
    assert (first_scores["action"] == "HUMAN_ESCALATION").sum() == floor(0.35 * len(first))
    assert first_scores["predicted_probability"].between(0, 1).all()


def test_demo_rejects_nonapproved_or_excluded_inputs() -> None:
    profile = inference.synthetic_portfolio(next(iter(inference.SCENARIOS.values())))

    with pytest.raises(ValueError, match="unexpected=.*default_next_month"):
        inference.score_portfolio(profile.assign(default_next_month=1))
    with pytest.raises(ValueError, match="excluded=.*SEX"):
        inference.score_portfolio(profile.assign(SEX=1))


def test_demo_inference_has_no_training_or_protected_data_access() -> None:
    source = inspect.getsource(inference).lower()

    for forbidden in (
        ".fit(",
        "final_eval",
        "final_hidden_evaluation",
        "load_research_inputs",
        "load_dataset",
        "read_parquet",
        "protected oracle",
    ):
        assert forbidden not in source
