import numpy as np
import pandas as pd

from autocollections.data.loader import normalize_dataset
from autocollections.data.schema import EXCLUDED_MODEL_FEATURES
from autocollections.features.approved_features import (
    DERIVED_MODEL_FEATURES,
    approved_derived_features,
    approved_feature_matrix,
    approved_feature_view,
)


def test_official_feature_view_excludes_demographics_and_is_deterministic() -> None:
    features = pd.DataFrame(
        {f"X{index}": [index * 100 + row for row in range(20)] for index in range(1, 24)}
    )
    target = pd.DataFrame({"Y": [row % 2 for row in range(20)]})
    frame = normalize_dataset(features, target, allow_subset=True)

    first = approved_feature_view(frame)
    second = approved_feature_view(frame)

    assert not set(EXCLUDED_MODEL_FEATURES) & set(first.columns)
    assert first.equals(second)
    assert first.shape == (20, 19)


def test_derived_features_are_finite_and_deterministic_with_zero_denominators() -> None:
    features = pd.DataFrame(
        {f"X{index}": [index * 100 + row for row in range(20)] for index in range(1, 24)}
    )
    target = pd.DataFrame({"Y": [row % 2 for row in range(20)]})
    frame = normalize_dataset(features, target, allow_subset=True)
    frame.loc[0, "LIMIT_BAL"] = 0
    frame.loc[0, [f"BILL_AMT{month}" for month in range(1, 7)]] = 0

    first = approved_derived_features(frame)
    second = approved_derived_features(frame)

    assert first.columns.tolist() == list(DERIVED_MODEL_FEATURES)
    assert np.isfinite(first.to_numpy()).all()
    assert first.equals(second)
    assert approved_feature_matrix(frame, include_derived=True).shape == (20, 32)
