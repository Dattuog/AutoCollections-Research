import json
from pathlib import Path

import yaml

from autocollections.data.loader import load_dataset
from autocollections.data.schema import ALLOWED_MODEL_FEATURES, EXCLUDED_MODEL_FEATURES, TARGET
from autocollections.data.splits import load_or_create_split_manifest
from autocollections.evaluation.protected_eval import (
    evaluate_candidate,
    evaluator_identity,
    load_research_inputs,
)
from autocollections.features.approved_features import approved_feature_view

ROOT = Path(__file__).resolve().parent
__all__ = ["evaluate_candidate", "evaluator_identity", "load_research_inputs"]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text())
    frame, metadata = load_dataset()
    approved = approved_feature_view(frame)
    approved_path = ROOT / "data/processed/approved_features.parquet"
    approved.to_parquet(approved_path, index=False)

    manifest = load_or_create_split_manifest(
        frame[TARGET],
        approved,
        ROOT / "data/processed/split_manifest.json",
        seed=config["seed"],
        fractions=config["split"],
        dataset_sha256=metadata["data_sha256"],
    )
    if manifest["split_sha256"] != config["canonical_split_sha256"]:
        raise ValueError("Split manifest does not match the canonical split hash")
    split_sizes = {name: len(indices) for name, indices in manifest["splits"].items()}
    identity = evaluator_identity()
    summary = {
        "dataset": "UCI 350",
        "rows": len(frame),
        "target_prevalence": float(frame[TARGET].mean()),
        "split_sizes": split_sizes,
        "allowed_raw_features": len(ALLOWED_MODEL_FEATURES),
        "excluded_demographic_features": list(EXCLUDED_MODEL_FEATURES),
        "dataset_sha256": metadata["data_sha256"],
        "split_sha256": manifest["split_sha256"],
        "split_strategy": manifest["strategy"],
        "evaluator_version": identity["evaluator_version"],
        "evaluator_sha256": identity["evaluator_sha256"],
        "strong_baseline_id": identity["strong_baseline_id"],
        "strong_baseline_utility_total": identity["strong_baseline_utility_total"],
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "evaluator_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    (reports / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print("AutoCollections prepare")
    print("-----------------------")
    print("Dataset: UCI 350")
    print(f"Rows: {len(frame)}")
    print(f"Target prevalence: {frame[TARGET].mean():.4f}")
    print(f"Train: {split_sizes['train']}")
    print(f"Validation: {split_sizes['validation']}")
    print(f"Hidden test: {split_sizes['hidden_test']}")
    print(f"Allowed raw features: {len(ALLOWED_MODEL_FEATURES)}")
    print(f"Excluded demographic features: {len(EXCLUDED_MODEL_FEATURES)}")
    print(f"Dataset hash: {metadata['data_sha256']}")
    print(f"Split hash: {manifest['split_sha256']}")
    print(f"Evaluator: {identity['evaluator_version']} ({identity['evaluator_sha256']})")
    print(
        f"Strong baseline: {identity['strong_baseline_id']} "
        f"({identity['strong_baseline_utility_total']:.2f})"
    )
    print("STATUS: READY (PROTECTED RESEARCH EVALUATOR)")


if __name__ == "__main__":
    main()
