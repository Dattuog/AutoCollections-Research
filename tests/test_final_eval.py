import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocollections.data.schema import ALLOWED_MODEL_FEATURES
from autocollections.evaluation import protected_eval
from scripts import final_eval

ROOT = Path(__file__).resolve().parents[1]


def test_selection_manifest_precedes_one_time_start_marker(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    marker = tmp_path / "started.json"
    output = tmp_path / "output.json"
    selection.write_text('{"selected_experiment":"exp_039"}\n')

    written = final_eval.mark_hidden_evaluation_started(selection, marker, output)

    assert marker.is_file()
    assert selection.stat().st_mtime_ns < marker.stat().st_mtime_ns
    assert written["selection_manifest_sha256"]
    with pytest.raises(ValueError, match="already started or completed"):
        final_eval.mark_hidden_evaluation_started(selection, marker, output)


def test_selected_candidate_and_manifest_hash_are_verified() -> None:
    selection = json.loads(final_eval.SELECTION_PATH.read_text())
    final_eval.verify_selected_candidate(selection)
    assert selection["selected_train_sha256"] == final_eval.sha256_file(ROOT / "train.py")


def test_selected_policy_is_deterministic_and_has_no_eval_target_input() -> None:
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        rng.integers(1, 1000, size=(80, len(ALLOWED_MODEL_FEATURES))),
        columns=ALLOWED_MODEL_FEATURES,
    )
    features["LIMIT_BAL"] += 10_000
    target = pd.Series(([0] * 40) + ([1] * 40))
    train = features.iloc[:60].copy()
    evaluation = features.iloc[60:].copy()
    train_before = train.copy()
    evaluation_before = evaluation.copy()

    first = final_eval.selected_candidate_predictions(train, target.iloc[:60], evaluation)
    second = final_eval.selected_candidate_predictions(train, target.iloc[:60], evaluation)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(evaluation, evaluation_before)
    assert tuple(inspect.signature(final_eval.selected_candidate_predictions).parameters) == (
        "X_train",
        "y_train",
        "X_eval",
    )


def test_only_explicit_final_path_indexes_hidden_split() -> None:
    normal_source = inspect.getsource(protected_eval.evaluate_candidate)
    final_source = inspect.getsource(final_eval.main)

    assert '["splits"]["hidden_test"]' not in normal_source
    assert '["splits"]["hidden_test"]' in final_source


def test_paired_bootstrap_is_deterministic() -> None:
    selected = np.array([2.0, 3.0, 4.0, 5.0])
    baseline = np.array([1.0, 1.0, 2.0, 3.0])
    first = final_eval.paired_bootstrap_interval(selected, baseline, resamples=100)
    second = final_eval.paired_bootstrap_interval(selected, baseline, resamples=100)

    assert first == second
    assert first["mean_difference"] == 1.75
