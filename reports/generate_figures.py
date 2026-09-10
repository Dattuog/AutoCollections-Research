import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autocollections-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
COLORS = {
    "navy": "#17324d",
    "blue": "#3f77a6",
    "teal": "#2a9d8f",
    "orange": "#e76f51",
    "gold": "#e9c46a",
    "gray": "#89939e",
}


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def research_trajectory(ledger: pd.DataFrame) -> None:
    experiment = ledger["experiment_id"].str.extract(r"(\d+)")[0].astype(int)
    score = pd.to_numeric(ledger["primary_score"], errors="coerce")
    eligible = ledger["feasible"].astype(str).eq("True") & score.gt(-1e17)
    cumulative_best = score.where(eligible).cummax().ffill().fillna(805.1444747208843)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1], sharex=True)
    axes[0].plot(experiment, cumulative_best, color=COLORS["navy"], linewidth=2.5)
    keep = ledger["status"].eq("KEEP")
    axes[0].scatter(
        experiment[keep], score[keep], color=COLORS["teal"], marker="o", s=35, label="KEEP"
    )
    axes[0].axhline(
        805.1444747208843,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
        label="B4 baseline",
    )
    axes[0].set(
        title="Best protected validation utility across 50 experiments", ylabel="Utility/account"
    )
    axes[0].legend(frameon=False, loc="lower right")

    status_style = {
        "KEEP": (COLORS["teal"], "o"),
        "REVERT": (COLORS["gray"], "x"),
        "CRASH": (COLORS["orange"], "s"),
        "INVALID": (COLORS["gold"], "D"),
        "TIMEOUT": (COLORS["orange"], "^"),
    }
    for status, (color, marker) in status_style.items():
        mask = ledger["status"].eq(status)
        if mask.any():
            axes[1].scatter(
                experiment[mask],
                np.zeros(mask.sum()),
                color=color,
                marker=marker,
                s=34,
                label=status,
            )
    axes[1].set(xlabel="Experiment", yticks=[], ylim=(-0.5, 0.5), title="KEEP / REVERT timeline")
    axes[1].legend(frameon=False, ncol=5, loc="lower center")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save("research_trajectory.png")


def final_performance(selection: dict, final: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    utility_labels = ["B4\nvalidation", "Final\nvalidation", "B4\nhidden", "Final\nhidden"]
    utility_values = [
        selection["strong_baseline_validation_score"],
        selection["selected_validation_score"],
        final["strong_baseline_b4"]["average_utility_per_account"],
        final["selected_candidate"]["average_utility_per_account"],
    ]
    bars = axes[0].bar(
        utility_labels,
        utility_values,
        color=[COLORS["gray"], COLORS["blue"], COLORS["gray"], COLORS["teal"]],
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set(title="Protected utility", ylabel="simulated_inr_units/account")

    actions = final["selected_candidate"]["action_distribution"]
    action_labels = ["No contact", "Digital", "Payment plan", "Human"]
    action_keys = ["NO_CONTACT", "DIGITAL_REMINDER", "PAYMENT_PLAN_REVIEW", "HUMAN_ESCALATION"]
    action_values = [actions[key]["count"] for key in action_keys]
    bars = axes[1].bar(
        action_labels,
        action_values,
        color=[COLORS["gray"], COLORS["blue"], COLORS["gold"], COLORS["orange"]],
    )
    axes[1].bar_label(bars, padding=3)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Final hidden action distribution", ylabel="Accounts")

    metrics = ["ROC-AUC", "PR-AUC", "Brier"]
    validation = selection["selected_validation_metrics"]
    hidden = final["selected_candidate"]
    x = np.arange(len(metrics))
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [validation["roc_auc"], validation["pr_auc"], validation["brier_score"]],
        width,
        label="Validation",
        color=COLORS["blue"],
    )
    axes[2].bar(
        x + width / 2,
        [hidden["roc_auc"], hidden["pr_auc"], hidden["brier_score"]],
        width,
        label="Hidden",
        color=COLORS["teal"],
    )
    axes[2].set(title="ML diagnostics", xticks=x, xticklabels=metrics, ylim=(0, 0.9))
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save("final_performance.png")


def utility_decomposition(final: dict) -> None:
    selected = final["selected_candidate"]
    baseline = final["strong_baseline_b4"]
    labels = [
        "Recovery",
        "Treatment cost",
        "Over-treatment",
        "Missed opportunity",
        "Policy penalty",
        "Net utility",
    ]

    def components(policy: dict) -> list[float]:
        return [
            policy["simulated_recovery_benefit_total"],
            -policy["treatment_cost_total"],
            -(
                policy["customer_experience_penalty_total"]
                + policy["unnecessary_escalation_penalty_total"]
            ),
            -policy["missed_opportunity_penalty_total"],
            -policy["policy_violation_penalty_total"],
            policy["net_business_utility"],
        ]

    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        x - width / 2,
        np.array(components(baseline)) / 1_000,
        width,
        label="B4 hidden",
        color=COLORS["gray"],
    )
    axis.bar(
        x + width / 2,
        np.array(components(selected)) / 1_000,
        width,
        label="Final hidden",
        color=COLORS["teal"],
    )
    axis.axhline(0, color=COLORS["navy"], linewidth=0.8)
    axis.set(
        title="Hidden-test utility decomposition: final candidate versus B4",
        ylabel="Thousands of simulated_inr_units",
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save("utility_decomposition.png")


def main() -> None:
    ledger = pd.read_csv(ROOT / "results.tsv", sep="\t")
    selection = json.loads((ROOT / "reports/final_selection_manifest.json").read_text())
    final = json.loads((ROOT / "reports/final_hidden_evaluation.json").read_text())
    research_trajectory(ledger)
    final_performance(selection, final)
    utility_decomposition(final)
    print("Wrote 3 final research figures from frozen artifacts")


if __name__ == "__main__":
    main()
