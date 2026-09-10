import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocollections.evaluation.protected_eval import (
    EVALUATOR_VERSION,
    STRONG_BASELINE_UTILITY_PER_ACCOUNT,
    evaluator_identity,
)
from autocollections.utils.hashing import sha256_bytes, sha256_file
from scripts.check_protected import MANIFEST_PATH, protected_hashes, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "results.tsv"
LATEST_SUMMARY = ROOT / "reports/experiments/latest_summary.json"
CANDIDATE_ARTIFACT = ROOT / "artifacts/candidates/latest/candidate_predictions.parquet"
LEDGER_FIELDS = (
    "experiment_id",
    "hypothesis",
    "description",
    "status",
    "primary_score",
    "feasible",
    "primary_score_eligible",
    "keep_eligible",
    "current_best_before",
    "best_improvement",
    "baseline_improvement_total",
    "recovery_total",
    "treatment_cost_total",
    "over_treatment_penalty_total",
    "missed_opportunity_penalty_total",
    "policy_penalty_total",
    "net_utility",
    "utility_per_account",
    "human_rate",
    "no_contact_count",
    "digital_count",
    "payment_plan_count",
    "human_count",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "runtime_seconds",
    "base_commit",
    "accepted_commit",
    "candidate_hash",
    "evaluator_version",
    "evaluator_hash",
    "artifact_hash",
    "error",
)
REQUIRED_RESULT_FIELDS = (
    "experiment_id",
    "primary_score",
    "feasible",
    "PRIMARY_SCORE_ELIGIBLE",
    "simulated_recovery_benefit_total",
    "treatment_cost_total",
    "over_treatment_penalty_total",
    "missed_opportunity_penalty_total",
    "policy_violation_penalty_total",
    "net_business_utility",
    "utility_per_account",
    "human_escalation_rate",
    "action_counts",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "runtime_seconds",
    "evaluator_version",
    "evaluator_sha256",
    "evaluation_split",
    "hidden_test_evaluated",
)
FORBIDDEN_SOURCE_TEXT = (
    "default_next_month",
    "hidden_test",
    "protected_oracle",
    "feasibility_audit",
    "policy_baselines_v2",
    "business_simulator",
    "protected_eval",
    "read_parquet",
    "read_json",
    "requests",
    "urllib",
    "httpx",
    "socket",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)


def preflight() -> tuple[str, dict[str, Any]]:
    head = _git("rev-parse", "HEAD").stdout.strip()
    if _git("branch", "--show-current").stdout.strip() != "autoresearch":
        raise RuntimeError("Phase 6 requires branch autoresearch")
    if not _git("tag", "--list", "phase5-baseline").stdout.strip():
        raise RuntimeError("Missing phase5-baseline tag")
    dirty = {
        line[3:]
        for line in _git("status", "--porcelain").stdout.splitlines()
        if line[3:] not in {"train.py", "results.tsv"}
    }
    if dirty:
        raise RuntimeError(f"Unexpected working-tree changes: {sorted(dirty)}")
    validate_ledger_header()
    manifest = json.loads(MANIFEST_PATH.read_text())
    if any(verify_manifest(manifest).values()):
        raise RuntimeError("Protected integrity preflight failed")
    identity = evaluator_identity()
    if identity["evaluator_version"] != EVALUATOR_VERSION:
        raise RuntimeError("Frozen evaluator version mismatch")
    return head, identity


def validate_candidate_source(source: str) -> None:
    found = [token for token in FORBIDDEN_SOURCE_TEXT if token in source]
    if found:
        raise ValueError(f"Forbidden candidate source references: {found}")
    if source.count("evaluate_candidate(") != 1:
        raise ValueError("Candidate must call the protected evaluator exactly once")


def validate_result(result: dict[str, Any], experiment_id: str, evaluator_hash: str) -> None:
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"Missing result fields: {missing}")
    if result["experiment_id"] != experiment_id:
        raise ValueError("Result experiment ID mismatch")
    if result["evaluator_version"] != EVALUATOR_VERSION:
        raise ValueError("Result evaluator version mismatch")
    if result["evaluator_sha256"] != evaluator_hash:
        raise ValueError("Result evaluator hash mismatch")
    if result["evaluation_split"] != "validation" or result["hidden_test_evaluated"] is not False:
        raise ValueError("Candidate result attempted non-validation evaluation")
    numeric = (
        "primary_score",
        "simulated_recovery_benefit_total",
        "treatment_cost_total",
        "over_treatment_penalty_total",
        "missed_opportunity_penalty_total",
        "policy_violation_penalty_total",
        "net_business_utility",
        "utility_per_account",
        "human_escalation_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "runtime_seconds",
    )
    if any(not math.isfinite(float(result[field])) for field in numeric):
        raise ValueError("Result contains non-finite metrics")
    if set(result["action_counts"]) != {
        "NO_CONTACT",
        "DIGITAL_REMINDER",
        "PAYMENT_PLAN_REVIEW",
        "HUMAN_ESCALATION",
    }:
        raise ValueError("Result action-count schema is invalid")


def decide(result: dict[str, Any] | None, best_score: float, protected_ok: bool) -> str:
    if not protected_ok:
        return "INVALID"
    if result is None:
        return "CRASH"
    if not result["feasible"] or not result["PRIMARY_SCORE_ELIGIBLE"]:
        return "REVERT"
    return "KEEP" if result["primary_score"] > best_score else "REVERT"


def validate_ledger_header(path: Path = LEDGER_PATH) -> None:
    header = tuple(path.open().readline().rstrip("\n").split("\t"))
    if header != LEDGER_FIELDS:
        raise ValueError("results.tsv header does not match the canonical ledger schema")


def best_score(path: Path = LEDGER_PATH) -> float:
    best = STRONG_BASELINE_UTILITY_PER_ACCOUNT
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] == "KEEP":
                best = max(best, float(row["primary_score"]))
    return best


def append_ledger(row: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LEDGER_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def restore_train(base_commit: str) -> None:
    _git("restore", f"--source={base_commit}", "--", "train.py")


def _ledger_row(
    *,
    experiment_id: str,
    hypothesis: str,
    description: str,
    status: str,
    base_commit: str,
    accepted_commit: str,
    candidate_hash: str,
    current_best: float,
    result: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    row = {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis.replace("\t", " ").replace("\n", " "),
        "description": description.replace("\t", " ").replace("\n", " "),
        "status": status,
        "base_commit": base_commit,
        "accepted_commit": accepted_commit,
        "candidate_hash": candidate_hash,
        "current_best_before": current_best,
        "error": error[:300].replace("\t", " ").replace("\n", " "),
    }
    if result is None:
        return row
    counts = result["action_counts"]
    row.update(
        {
            "primary_score": result["primary_score"],
            "feasible": result["feasible"],
            "primary_score_eligible": result["PRIMARY_SCORE_ELIGIBLE"],
            "keep_eligible": status == "KEEP",
            "best_improvement": result["primary_score"] - current_best,
            "baseline_improvement_total": result["baseline_improvement_total"],
            "recovery_total": result["simulated_recovery_benefit_total"],
            "treatment_cost_total": result["treatment_cost_total"],
            "over_treatment_penalty_total": result["over_treatment_penalty_total"],
            "missed_opportunity_penalty_total": result["missed_opportunity_penalty_total"],
            "policy_penalty_total": result["policy_violation_penalty_total"],
            "net_utility": result["net_business_utility"],
            "utility_per_account": result["utility_per_account"],
            "human_rate": result["human_escalation_rate"],
            "no_contact_count": counts["NO_CONTACT"],
            "digital_count": counts["DIGITAL_REMINDER"],
            "payment_plan_count": counts["PAYMENT_PLAN_REVIEW"],
            "human_count": counts["HUMAN_ESCALATION"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "brier_score": result["brier_score"],
            "runtime_seconds": result["runtime_seconds"],
            "evaluator_version": result["evaluator_version"],
            "evaluator_hash": result["evaluator_sha256"],
            "artifact_hash": (
                sha256_file(CANDIDATE_ARTIFACT) if CANDIDATE_ARTIFACT.exists() else ""
            ),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one protected AutoCollections experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    base_commit, identity = preflight()
    current_best = best_score()
    candidate_source = (ROOT / "train.py").read_text()
    candidate_hash = sha256_bytes(candidate_source.encode())
    manifest_bytes = MANIFEST_PATH.read_bytes()
    before_hashes = protected_hashes()
    result = None
    error = ""
    status = "CRASH"
    accepted_commit = ""
    output = ""
    LATEST_SUMMARY.unlink(missing_ok=True)

    try:
        validate_candidate_source(candidate_source)
        environment = {
            "PATH": os.environ["PATH"],
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "AUTOCOLLECTIONS_EXPERIMENT_ID": args.experiment_id,
        }
        completed = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"candidate exited {completed.returncode}")
        result = json.loads(LATEST_SUMMARY.read_text())
        validate_result(result, args.experiment_id, identity["evaluator_sha256"])
    except subprocess.TimeoutExpired as failure:
        status = "TIMEOUT"
        output = (failure.stdout or "") + (failure.stderr or "")
        error = f"candidate timed out after {args.timeout}s"
    except (KeyError, TypeError, ValueError) as failure:
        status = "INVALID"
        error = str(failure)
    except (OSError, RuntimeError) as failure:
        error = str(failure)

    report_dir = ROOT / "reports/experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "run.log").write_text(output)
    (report_dir / f"{args.experiment_id}.log").write_text(output)
    protected_ok = (
        manifest_bytes == MANIFEST_PATH.read_bytes()
        and before_hashes == protected_hashes()
        and not any(verify_manifest(json.loads(manifest_bytes)).values())
    )
    if status != "TIMEOUT":
        status = decide(result, current_best, protected_ok)

    if status == "KEEP":
        _git("add", "train.py")
        message = f"research: {args.experiment_id} {args.description}"[:100]
        _git("commit", "-m", message)
        accepted_commit = _git("rev-parse", "HEAD").stdout.strip()
    else:
        restore_train(base_commit)

    row = _ledger_row(
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        description=args.description,
        status=status,
        base_commit=base_commit,
        accepted_commit=accepted_commit,
        candidate_hash=candidate_hash,
        current_best=current_best,
        result=result,
        error=error,
    )
    append_ledger(row)
    experiment_report = {
        "runner_version": 1,
        "decision": status,
        "base_commit": base_commit,
        "accepted_commit": accepted_commit,
        "candidate_train_sha256": candidate_hash,
        "protected_integrity": "PASS" if protected_ok else "FAIL",
        "hypothesis": args.hypothesis,
        "description": args.description,
        "error": error,
        "result": result,
    }
    report_path = report_dir / f"{args.experiment_id}.json"
    report_path.write_text(json.dumps(experiment_report, indent=2, sort_keys=True) + "\n")
    print(f"experiment_id: {args.experiment_id}")
    print(f"decision: {status}")
    print(f"base_commit: {base_commit}")
    print(f"accepted_commit: {accepted_commit or '-'}")
    print(f"protected_integrity: {'PASS' if protected_ok else 'FAIL'}")
    print(f"best_before: {current_best}")
    if result is not None:
        print(f"candidate_score: {result['primary_score']}")
        print(f"feasible: {result['feasible']}")
    if error:
        print(f"error: {error}")


if __name__ == "__main__":
    main()
