import subprocess
from pathlib import Path

import pytest

from autocollections.evaluation.protected_eval import STRONG_BASELINE_UTILITY_PER_ACCOUNT
from scripts import run_experiment


def valid_result(score: float = 900.0, feasible: bool = True) -> dict:
    return {
        "experiment_id": "exp_001",
        "primary_score": score,
        "feasible": feasible,
        "PRIMARY_SCORE_ELIGIBLE": feasible,
        "simulated_recovery_benefit_total": 1.0,
        "treatment_cost_total": 1.0,
        "over_treatment_penalty_total": 1.0,
        "missed_opportunity_penalty_total": 1.0,
        "policy_violation_penalty_total": 0.0,
        "net_business_utility": 1.0,
        "utility_per_account": score,
        "human_escalation_rate": 0.1,
        "action_counts": {
            "NO_CONTACT": 1,
            "DIGITAL_REMINDER": 1,
            "PAYMENT_PLAN_REVIEW": 1,
            "HUMAN_ESCALATION": 1,
        },
        "roc_auc": 0.7,
        "pr_auc": 0.5,
        "brier_score": 0.2,
        "runtime_seconds": 1.0,
        "evaluator_version": run_experiment.EVALUATOR_VERSION,
        "evaluator_sha256": "evaluator",
        "evaluation_split": "validation",
        "hidden_test_evaluated": False,
    }


def test_keep_revert_infeasible_crash_and_integrity_decisions() -> None:
    assert run_experiment.decide(valid_result(900), 800, True) == "KEEP"
    assert run_experiment.decide(valid_result(800), 800, True) == "REVERT"
    assert run_experiment.decide(valid_result(900, feasible=False), 800, True) == "REVERT"
    assert run_experiment.decide(valid_result(900), 800, False) == "INVALID"
    assert run_experiment.decide(None, 800, True) == "CRASH"


def test_result_and_candidate_source_validation() -> None:
    result = valid_result()
    run_experiment.validate_result(result, "exp_001", "evaluator")
    result["primary_score"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        run_experiment.validate_result(result, "exp_001", "evaluator")

    run_experiment.validate_candidate_source("evaluate_candidate(candidate)\n")
    with pytest.raises(ValueError, match="Forbidden"):
        run_experiment.validate_candidate_source(
            "hidden_test = True\nevaluate_candidate(candidate)\n"
        )


def test_ledger_append_and_best_score_tracking(tmp_path: Path) -> None:
    ledger = tmp_path / "results.tsv"
    ledger.write_text("\t".join(run_experiment.LEDGER_FIELDS) + "\n")
    run_experiment.validate_ledger_header(ledger)
    run_experiment.append_ledger(
        {"experiment_id": "exp_001", "status": "KEEP", "primary_score": 810.0},
        ledger,
    )
    run_experiment.append_ledger(
        {"experiment_id": "exp_002", "status": "REVERT", "primary_score": 900.0},
        ledger,
    )

    assert len(ledger.read_text().splitlines()) == 3
    assert run_experiment.best_score(ledger) == 810.0
    empty = tmp_path / "empty.tsv"
    empty.write_text("\t".join(run_experiment.LEDGER_FIELDS) + "\n")
    assert run_experiment.best_score(empty) == STRONG_BASELINE_UTILITY_PER_ACCOUNT
    empty.write_text("mutable\theader\n")
    with pytest.raises(ValueError, match="canonical ledger schema"):
        run_experiment.validate_ledger_header(empty)


def test_crash_recovery_restores_only_train(tmp_path: Path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "train.py").write_text("accepted = True\n")
    (tmp_path / "results.tsv").write_text("ledger\n")
    subprocess.run(["git", "add", "train.py", "results.tsv"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "train.py").write_text("this crashes\n")
    (tmp_path / "results.tsv").write_text("ledger preserved\n")
    monkeypatch.setattr(run_experiment, "ROOT", tmp_path)

    run_experiment.restore_train(base_commit)

    assert (tmp_path / "train.py").read_text() == "accepted = True\n"
    assert (tmp_path / "results.tsv").read_text() == "ledger preserved\n"
