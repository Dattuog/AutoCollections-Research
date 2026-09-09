import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autocollections-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocollections.data.loader import load_dataset
from autocollections.data.schema import (
    ALL_FEATURES,
    ALLOWED_MODEL_FEATURES,
    EXCLUDED_MODEL_FEATURES,
    TARGET,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"


def _save_current_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150)
    plt.close()


def main() -> None:
    frame, metadata = load_dataset()
    FIGURES.mkdir(parents=True, exist_ok=True)

    numeric_ranges = {
        column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
        for column in frame.columns
    }
    suspicious = {
        "EDUCATION_outside_1_to_4": sorted(
            int(value) for value in set(frame["EDUCATION"]) - {1, 2, 3, 4}
        ),
        "MARRIAGE_outside_1_to_3": sorted(
            int(value) for value in set(frame["MARRIAGE"]) - {1, 2, 3}
        ),
        "repayment_status_outside_minus_2_to_8": {
            column: sorted(int(value) for value in set(frame[column]) if value < -2 or value > 8)
            for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
        },
    }
    demographic_summary = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "unique_values": int(frame[column].nunique()),
        }
        for column in EXCLUDED_MODEL_FEATURES
    }
    financial_summary = {
        column: {
            key: float(value)
            for key, value in frame[column].describe()[["mean", "std", "min", "50%", "max"]].items()
        }
        for column in ALLOWED_MODEL_FEATURES
    }
    duplicate_summary = {
        "complete_23_features_plus_target": int(frame.duplicated().sum()),
        "all_23_features_only": int(frame.duplicated(list(ALL_FEATURES)).sum()),
        "approved_19_model_features_only": int(
            frame.duplicated(list(ALLOWED_MODEL_FEATURES)).sum()
        ),
    }
    summary = {
        "dataset": "UCI 350",
        "dataset_sha256": metadata["data_sha256"],
        "row_count": len(frame),
        "feature_count": len(frame.columns) - 1,
        "target_prevalence": float(frame[TARGET].mean()),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts().sort_index().items()
        },
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_rows_beyond_first": duplicate_summary,
        "numeric_ranges": numeric_ranges,
        "suspicious_categorical_values": suspicious,
        "excluded_demographic_summary": demographic_summary,
        "financial_feature_distributions": financial_summary,
        "notes": [
            "Unusual categorical values are reported, not removed.",
            "Demographic columns are summarized for data quality only and excluded from the official model view.",
            "Utilization is clipped to [0, 5] for plot readability only; prepared values are unchanged.",
        ],
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "eda_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    frame[TARGET].value_counts().sort_index().plot.bar(color=["#4c78a8", "#f58518"])
    plt.title("Default next month target distribution")
    plt.xlabel("Default next month")
    plt.ylabel("Accounts")
    _save_current_figure("target_distribution.png")

    repayment = frame[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].melt(
        var_name="month", value_name="status"
    )
    repayment.groupby(["status", "month"]).size().unstack(fill_value=0).plot.bar(figsize=(9, 5))
    plt.title("Repayment status distribution")
    plt.xlabel("Repayment status")
    plt.ylabel("Observations")
    _save_current_figure("repayment_status_distribution.png")

    utilization = (frame["BILL_AMT1"].clip(lower=0) / frame["LIMIT_BAL"].clip(lower=1)).clip(
        upper=5
    )
    utilization.plot.hist(bins=60, color="#4c78a8")
    plt.title("Latest utilization distribution (plot clipped at 5)")
    plt.xlabel("BILL_AMT1 / LIMIT_BAL")
    _save_current_figure("utilization_distribution.png")

    sample = frame.sample(n=min(3_000, len(frame)), random_state=42)
    plt.scatter(sample["BILL_AMT1"], sample["PAY_AMT1"], s=7, alpha=0.25)
    plt.title("Latest bill amount vs latest payment")
    plt.xlabel("BILL_AMT1")
    plt.ylabel("PAY_AMT1")
    _save_current_figure("payment_bill_relationship.png")

    print(f"EDA rows: {len(frame)}")
    print(f"Target prevalence: {frame[TARGET].mean():.4f}")
    print("Wrote reports/eda_summary.json and 4 figures")


if __name__ == "__main__":
    main()
