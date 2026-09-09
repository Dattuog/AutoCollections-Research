import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autocollections.data.schema import (
    TARGET,
    normalize_feature_columns,
    normalize_name,
    normalize_target,
    validate_processed_dataset,
)
from autocollections.utils.hashing import sha256_dataframe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "data/processed/uci_credit_default.parquet"
DEFAULT_METADATA_PATH = ROOT / "data/processed/dataset_metadata.json"
DEFAULT_MANUAL_PATH = ROOT / "data/raw/default_of_credit_card_clients.xls"


def normalize_dataset(
    features: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    *,
    allow_subset: bool = False,
) -> pd.DataFrame:
    normalized_features = normalize_feature_columns(features).reset_index(drop=True)
    normalized_target = normalize_target(target)
    if len(normalized_features) != len(normalized_target):
        raise ValueError("Feature and target row counts differ")

    frame = normalized_features.assign(**{TARGET: normalized_target})
    validate_processed_dataset(frame, allow_subset=allow_subset)
    return frame


def _fetch_uci() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    return dataset.data.features.copy(), dataset.data.targets.copy(), "ucimlrepo"


def _load_manual(path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    last_columns: list[str] = []
    for header in (1, 0):
        frame = pd.read_excel(path, header=header)
        frame.columns = [normalize_name(column) for column in frame.columns]
        last_columns = frame.columns.tolist()
        if TARGET in frame:
            target = frame.pop(TARGET)
            return frame, target, f"manual:{path.name}"
    raise ValueError(f"Could not find target column in manual workbook; columns={last_columns}")


def _metadata(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "uci_dataset_id": 350,
        "license": "CC BY 4.0",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "feature_count": len(frame.columns) - 1,
        "target": TARGET,
        "data_sha256": sha256_dataframe(frame),
    }


def load_dataset(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    manual_path: Path = DEFAULT_MANUAL_PATH,
    allow_subset: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        validate_processed_dataset(frame, allow_subset=allow_subset)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = _metadata(frame, "cache")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        actual_hash = sha256_dataframe(frame)
        if metadata.get("data_sha256") != actual_hash:
            raise ValueError("Cached dataset hash does not match dataset metadata")
        return frame, metadata

    try:
        features, target, source = _fetch_uci()
    except Exception as error:
        if not manual_path.exists():
            raise RuntimeError(
                f"UCI retrieval failed and manual fallback was not found at {manual_path}"
            ) from error
        features, target, source = _load_manual(manual_path)

    frame = normalize_dataset(features, target, allow_subset=allow_subset)
    metadata = _metadata(frame, source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return frame, metadata
