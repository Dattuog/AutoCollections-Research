from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def risk_metric_report(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    y_true = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probability):
        raise ValueError("Target and probability row counts differ")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1]")

    prediction = (probability >= threshold).astype(int)
    quantiles = np.quantile(probability, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    bin_ids = np.digitize(probability, np.linspace(0, 1, calibration_bins + 1)[1:-1])
    bins = []
    calibration_error = 0.0
    for bin_id in range(calibration_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        mean_probability = float(probability[mask].mean())
        observed_rate = float(y_true[mask].mean())
        count = int(mask.sum())
        calibration_error += count / len(y_true) * abs(mean_probability - observed_rate)
        bins.append(
            {
                "bin": bin_id,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_default_rate": observed_rate,
            }
        )

    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - probability, probability]))),
        "precision_at_0_5": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix_at_0_5": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "predicted_probability_distribution": {
            "min": float(probability.min()),
            "max": float(probability.max()),
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "p50": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
        },
        "calibration": {
            "binning": "uniform",
            "requested_bins": calibration_bins,
            "expected_calibration_error": float(calibration_error),
            "bins": bins,
        },
    }
