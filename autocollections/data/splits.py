import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autocollections.utils.hashing import sha256_json

SPLIT_NAMES = ("train", "validation", "hidden_test")
SPLIT_STRATEGY = "stratified_grouped_approved_features_v1"


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256_json({key: value for key, value in payload.items() if key != "split_sha256"})


def _feature_groups(group_features: pd.DataFrame) -> list[list[int]]:
    return [
        [int(sample_id) for sample_id in sample_ids]
        for sample_ids in group_features.groupby(
            list(group_features.columns), sort=False, dropna=False
        ).indices.values()
    ]


def validate_split_manifest(manifest: dict[str, Any], group_features: pd.DataFrame) -> None:
    if manifest.get("split_sha256") != _payload_hash(manifest):
        raise ValueError("Split manifest hash is invalid")
    if manifest.get("strategy") != SPLIT_STRATEGY:
        raise ValueError("Split manifest does not use the required grouped strategy")
    if manifest.get("group_columns") != group_features.columns.tolist():
        raise ValueError("Split manifest grouping columns do not match the approved feature view")

    splits = manifest.get("splits", {})
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Split manifest must contain train, validation, and hidden_test")
    all_ids = [sample_id for values in splits.values() for sample_id in values]
    row_count = len(group_features)
    if len(all_ids) != row_count or len(set(all_ids)) != row_count:
        raise ValueError("Split rows must be complete and non-overlapping")
    if set(all_ids) != set(range(row_count)):
        raise ValueError("Split manifest sample IDs do not match dataset rows")

    split_by_id = {sample_id: name for name, ids in splits.items() for sample_id in ids}
    if any(
        len({split_by_id[sample_id] for sample_id in group}) > 1
        for group in _feature_groups(group_features)
    ):
        raise ValueError("Identical approved feature rows cross split boundaries")


def create_split_manifest(
    target: pd.Series,
    group_features: pd.DataFrame,
    *,
    seed: int,
    fractions: dict[str, float],
    dataset_sha256: str,
) -> dict[str, Any]:
    if set(fractions) != set(SPLIT_NAMES) or abs(sum(fractions.values()) - 1.0) > 1e-12:
        raise ValueError("Split fractions must contain train/validation/hidden_test and sum to 1")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("Split fractions must be positive")
    if len(target) != len(group_features):
        raise ValueError("Target and grouping feature row counts differ")

    target = target.reset_index(drop=True)
    group_features = group_features.reset_index(drop=True)
    sample_ids = np.arange(len(target))
    train_ids, remainder_ids = train_test_split(
        sample_ids,
        train_size=fractions["train"],
        random_state=seed,
        stratify=target,
    )
    remainder_target = target.iloc[remainder_ids]
    validation_share = fractions["validation"] / (
        fractions["validation"] + fractions["hidden_test"]
    )
    validation_ids, hidden_test_ids = train_test_split(
        remainder_ids,
        train_size=validation_share,
        random_state=seed,
        stratify=remainder_target,
    )
    reference = dict(zip(SPLIT_NAMES, (train_ids, validation_ids, hidden_test_ids), strict=True))
    desired_sizes = {name: len(ids) for name, ids in reference.items()}
    desired_classes = {
        name: {label: int((target.iloc[ids] == label).sum()) for label in (0, 1)}
        for name, ids in reference.items()
    }

    groups = _feature_groups(group_features)
    duplicate_groups = [group for group in groups if len(group) > 1]
    singleton_ids = [group[0] for group in groups if len(group) == 1]
    rng = np.random.default_rng(seed)
    rng.shuffle(duplicate_groups)
    duplicate_groups.sort(key=len, reverse=True)

    assigned: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    assigned_classes = {name: {0: 0, 1: 0} for name in SPLIT_NAMES}
    for group in duplicate_groups:
        group_classes = {label: int((target.iloc[group] == label).sum()) for label in (0, 1)}
        viable = [
            name
            for name in SPLIT_NAMES
            if len(assigned[name]) + len(group) <= desired_sizes[name]
            and all(
                assigned_classes[name][label] + group_classes[label] <= desired_classes[name][label]
                for label in (0, 1)
            )
        ]
        if not viable:
            raise ValueError("Feature groups cannot fit the configured stratified split quotas")

        # ponytail: greedy placement fits the current small duplicate groups; use an
        # optimizer only if future datasets make exact quotas infeasible.
        name = min(
            viable,
            key=lambda candidate: (
                max(
                    (len(assigned[candidate]) + len(group)) / desired_sizes[candidate],
                    *(
                        (assigned_classes[candidate][label] + group_classes[label])
                        / desired_classes[candidate][label]
                        for label in (0, 1)
                    ),
                ),
                SPLIT_NAMES.index(candidate),
            ),
        )
        assigned[name].extend(group)
        for label in (0, 1):
            assigned_classes[name][label] += group_classes[label]

    for label in (0, 1):
        pool = np.array(
            [sample_id for sample_id in singleton_ids if target.iloc[sample_id] == label]
        )
        rng.shuffle(pool)
        offset = 0
        for name in SPLIT_NAMES:
            needed = desired_classes[name][label] - assigned_classes[name][label]
            assigned[name].extend(int(sample_id) for sample_id in pool[offset : offset + needed])
            assigned_classes[name][label] += needed
            offset += needed
        if offset != len(pool):
            raise ValueError("Singleton rows did not fill the configured stratified split quotas")

    manifest: dict[str, Any] = {
        "version": 1,
        "strategy": SPLIT_STRATEGY,
        "group_columns": group_features.columns.tolist(),
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "fractions": fractions,
        "splits": {name: sorted(assigned[name]) for name in SPLIT_NAMES},
    }
    manifest["split_sha256"] = _payload_hash(manifest)
    validate_split_manifest(manifest, group_features)
    return manifest


def load_or_create_split_manifest(
    target: pd.Series,
    group_features: pd.DataFrame,
    path: Path,
    *,
    seed: int,
    fractions: dict[str, float],
    dataset_sha256: str,
) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_split_manifest(manifest, group_features)
        if manifest["dataset_sha256"] != dataset_sha256:
            raise ValueError("Existing split manifest belongs to a different processed dataset")
        if manifest["seed"] != seed or manifest["fractions"] != fractions:
            raise ValueError("Existing split manifest does not match benchmark configuration")
        return manifest

    manifest = create_split_manifest(
        target,
        group_features,
        seed=seed,
        fractions=fractions,
        dataset_sha256=dataset_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
