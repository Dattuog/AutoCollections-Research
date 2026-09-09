from pathlib import Path

import pandas as pd
import pytest
import yaml

from autocollections.data import loader
from autocollections.data.schema import ALL_FEATURES, TARGET
from autocollections.data.splits import (
    create_split_manifest,
    load_or_create_split_manifest,
    validate_split_manifest,
)
from autocollections.utils.hashing import sha256_dataframe

FRACTIONS = {"train": 0.70, "validation": 0.15, "hidden_test": 0.15}
ROOT = Path(__file__).resolve().parents[1]


def fixture_data(rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {f"X{index}": [index * 100 + row for row in range(rows)] for index in range(1, 24)}
    )
    target = pd.DataFrame({"Y": [row % 2 for row in range(rows)]})
    return features, target


def test_benchmark_configuration() -> None:
    benchmark = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())

    assert benchmark["seed"] == 42
    assert (
        benchmark["canonical_split_sha256"]
        == "6d8f4e9a9355cb49351af5fb83d6a6a938538d47c084fd039358cbed9ae0e0ec"
    )
    assert benchmark["split"] == FRACTIONS
    assert (ROOT / "configs/business_simulation.yaml").is_file()


def test_dataset_normalization_and_hash_are_deterministic() -> None:
    features, target = fixture_data()
    first = loader.normalize_dataset(features, target, allow_subset=True)
    second = loader.normalize_dataset(features, target, allow_subset=True)

    assert first.columns.tolist() == [*ALL_FEATURES, TARGET]
    assert TARGET not in first.loc[:, ALL_FEATURES]
    assert not first[TARGET].isna().any()
    assert set(first[TARGET]) == {0, 1}
    assert sha256_dataframe(first) == sha256_dataframe(second)


def test_full_mode_requires_the_complete_dataset() -> None:
    features, target = fixture_data()

    with pytest.raises(ValueError, match="Expected 30000 rows"):
        loader.normalize_dataset(features, target)


def test_loader_fetches_then_uses_verified_cache(tmp_path: Path, monkeypatch) -> None:
    features, target = fixture_data()
    monkeypatch.setattr(loader, "_fetch_uci", lambda: (features, target, "test-fixture"))
    cache = tmp_path / "dataset.parquet"
    metadata = tmp_path / "metadata.json"

    first, first_metadata = loader.load_dataset(
        cache_path=cache,
        metadata_path=metadata,
        manual_path=tmp_path / "missing.xls",
        allow_subset=True,
    )
    monkeypatch.setattr(loader, "_fetch_uci", lambda: (_ for _ in ()).throw(AssertionError()))
    second, second_metadata = loader.load_dataset(
        cache_path=cache,
        metadata_path=metadata,
        manual_path=tmp_path / "missing.xls",
        allow_subset=True,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_metadata == second_metadata
    assert first_metadata["data_sha256"] == sha256_dataframe(first)


def test_split_manifest_is_deterministic_stratified_and_complete(tmp_path: Path) -> None:
    target = pd.Series(([0] * 50) + ([1] * 50))
    group_features = pd.DataFrame({"feature": range(100)})
    group_features.loc[50, "feature"] = group_features.loc[0, "feature"]
    first = create_split_manifest(
        target,
        group_features,
        seed=42,
        fractions=FRACTIONS,
        dataset_sha256="abc",
    )
    second = load_or_create_split_manifest(
        target,
        group_features,
        tmp_path / "split_manifest.json",
        seed=42,
        fractions=FRACTIONS,
        dataset_sha256="abc",
    )

    assert first == second
    assert {name: len(ids) for name, ids in first["splits"].items()} == {
        "train": 70,
        "validation": 15,
        "hidden_test": 15,
    }
    assert "target" not in (tmp_path / "split_manifest.json").read_text().lower()
    for ids in first["splits"].values():
        assert 0.4 <= target.iloc[ids].mean() <= 0.6
    split_by_id = {
        sample_id: name for name, sample_ids in first["splits"].items() for sample_id in sample_ids
    }
    assert split_by_id[0] == split_by_id[50]
    validate_split_manifest(first, group_features)
