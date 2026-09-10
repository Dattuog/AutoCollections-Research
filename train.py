import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from autocollections.features.approved_features import approved_feature_matrix
from prepare import evaluate_candidate, load_research_inputs

SEED = 42
DIGITAL_REMINDER = 1
HUMAN_ESCALATION = 3
HUMAN_CAPACITY = 0.35


def main() -> None:
    started = perf_counter()
    X_train, y_train, X_validation, validation_ids = load_research_inputs()
    train_features = approved_feature_matrix(X_train, include_derived=True)
    validation_features = approved_feature_matrix(X_validation, include_derived=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=100,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(train_features, y_train)
    probabilities = model.predict_proba(validation_features)[:, 1]

    actions = np.full(len(probabilities), DIGITAL_REMINDER, dtype=int)
    human_count = int(np.floor(HUMAN_CAPACITY * len(probabilities)))
    exposure = X_validation["BILL_AMT1"].clip(lower=0).to_numpy()
    utilization = exposure / X_validation["LIMIT_BAL"].clip(lower=1).to_numpy()
    allocation_score = probabilities * np.power(exposure, 0.60) * (1 + utilization)
    actions[np.argsort(-allocation_score, kind="stable")[:human_count]] = HUMAN_ESCALATION
    candidate = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "predicted_probability": probabilities,
            "action": actions,
        }
    )
    artifact_dir = Path("artifacts/candidates/latest")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_parquet(artifact_dir / "candidate_predictions.parquet", index=False)

    result = evaluate_candidate(
        candidate,
        experiment_id=os.environ.get("AUTOCOLLECTIONS_EXPERIMENT_ID", "phase5_b4_dry_run"),
        runtime_seconds=perf_counter() - started,
    )
    print("--- AUTOCOLLECTIONS_RESULT ---")
    for key in (
        "experiment_id",
        "primary_score",
        "feasible",
        "beats_strong_baseline",
        "protected_business_utility",
        "utility_per_account",
        "human_escalation_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "runtime_seconds",
        "evaluator_version",
        "evaluator_sha256",
    ):
        print(f"{key}: {result[key]}")
    print("--- END_RESULT ---")


if __name__ == "__main__":
    main()
