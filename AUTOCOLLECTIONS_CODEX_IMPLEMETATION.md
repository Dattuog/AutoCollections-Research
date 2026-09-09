# AutoCollections Research — Complete Codex Implementation Specification

> **Purpose of this file:** This is the implementation contract for Codex. Read it completely before modifying the repository. Implement the project in phases and satisfy each phase's acceptance criteria before moving on.
>
> **Project:** AutoCollections Research — Autonomous Next-Best-Action Optimization for Financial Operations
>
> **Core idea:** Adapt Andrej Karpathy's AutoResearch pattern to a business-oriented collections/customer-treatment problem. An AI research agent may modify only the candidate policy/model implementation, while data preparation, evaluation, business simulation, constraints, tests, and final holdout evaluation remain protected.

---

## 0. Non-Negotiable Project Principles

1. **Do not build a generic AutoML project.** Preserve the AutoResearch pattern:
   - `prepare.py` = fixed/protected data preparation and evaluation utilities.
   - `train.py` = the single research file an agent may edit.
   - `program.md` = human-authored research charter.
   - `results.tsv` = append-only experiment ledger.
   - Git = accepted experiments are committed; failed experiments are reverted.
   - Every candidate receives the same experiment budget and evaluation protocol.

2. **Business outcome first.** The research objective is not raw classification accuracy. The primary objective is **simulated business utility** under quality, cost, operational, and policy constraints.

3. **No fabricated real-world claims.** The UCI dataset does not contain historical treatment outcomes. Recovery amounts, contact costs, and treatment effects in Phase 1 are a **reproducible simulation**. Never describe them as actual bank savings or real intervention effects.

4. **No real PII and no real customer decisions.** This project is a research/portfolio system using public data and simulated treatment actions.

5. **Exclude demographic attributes from treatment modeling by default.** `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` must not be used as predictive/treatment-policy features in the official benchmark. They may be retained only for data-quality analysis if needed. This is a conservative responsible-AI design choice.

6. **Hidden test is not an optimization signal.** During AutoResearch, the agent may receive aggregate validation metrics only. The final hidden test is evaluated once after the selected candidate is frozen.

7. **Prefer simple models first.** Logistic regression + calibrated probabilities + interpretable policy thresholds must exist before optional tree ensembles.

8. **Do not add LLM negotiation, DebtBench, FastAPI, Streamlit, Docker, Kubernetes, Kafka, Redis, or cloud infrastructure until the core AutoResearch loop works correctly.** Productization is Phase 7; the GenAI extension is Phase 8.

---

# 1. Business Problem

A financial-operations team has a set of accounts with different repayment-risk profiles. Treating every account identically is inefficient:

- contacting everyone creates unnecessary operational cost and poor customer experience;
- contacting nobody creates missed recovery opportunities;
- escalating too many accounts to human review is expensive;
- a fixed threshold may become suboptimal as model features and calibration change.

The system should estimate account risk and select an allowed next action.

## Initial action space

Use exactly these four actions in the MVP:

```python
NO_CONTACT = 0
DIGITAL_REMINDER = 1
PAYMENT_PLAN_REVIEW = 2
HUMAN_REVIEW = 3
```

Interpret them conservatively:

| Action | Meaning |
|---|---|
| `NO_CONTACT` | No proactive treatment for now. |
| `DIGITAL_REMINDER` | Low-cost generic digital reminder / self-service prompt. |
| `PAYMENT_PLAN_REVIEW` | Route for an approved payment-plan / affordability review workflow. |
| `HUMAN_REVIEW` | Escalate to a trained human for complex/high-risk cases. |

The MVP does **not** generate collection messages and does **not** negotiate with customers.

---

# 2. Dataset

## Primary dataset

Use the UCI Machine Learning Repository dataset:

**Default of Credit Card Clients — UCI dataset ID 350**

Dataset facts relevant to implementation:

- 30,000 instances.
- 23 predictive features.
- Binary default-payment target.
- Financial/payment-history variables.
- CC BY 4.0 license.

### Preferred acquisition method

Use `ucimlrepo` for reproducibility rather than committing the raw `.xls` file to Git.

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=350)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
```

If network retrieval fails, support a manual fallback file at:

```text
data/raw/default_of_credit_card_clients.xls
```

Do not commit the raw dataset by default. Add raw/processed data outputs to `.gitignore`.

## Target semantics

Rename the target to:

```text
default_next_month
```

Normalize it to integer `{0, 1}` where:

- `0` = did not default next month;
- `1` = defaulted next month.

The model predicts:

```text
P(default_next_month = 1 | approved financial/behavioral features)
```

Do not describe this probability as a causal treatment-response probability.

---

# 3. Repository Structure

Codex should create the following structure. Do not create extra services unless required later by this specification.

```text
AutoCollections-Research/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── program.md
├── prepare.py
├── train.py
├── results.tsv
│
├── configs/
│   ├── benchmark.yaml
│   └── business_simulation.yaml
│
├── autocollections/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── splits.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── approved_features.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   └── business_simulator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── objective.py
│   │   ├── baselines.py
│   │   └── protected_eval.py
│   ├── artifacts/
│   │   ├── __init__.py
│   │   └── manifest.py
│   └── utils/
│       ├── __init__.py
│       ├── hashing.py
│       └── seed.py
│
├── scripts/
│   ├── eda.py
│   ├── run_experiment.py
│   ├── check_protected.py
│   ├── final_eval.py
│   └── leaderboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── candidates/
│   └── approved/
│
├── reports/
│   ├── figures/
│   └── experiments/
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_simulator.py
    ├── test_objective.py
    ├── test_baselines.py
    ├── test_train_contract.py
    ├── test_protected_integrity.py
    └── test_reproducibility.py
```

---

# 4. Python Environment

Use Python 3.12 and `uv`.

## `pyproject.toml`

Core dependencies:

```text
pandas
numpy
scikit-learn
pyarrow
pyyaml
joblib
ucimlrepo
matplotlib
```

Development dependencies:

```text
pytest
pytest-cov
ruff
```

Optional only after the classical baseline is complete:

```text
lightgbm
```

Do not add an LLM SDK in the MVP.

## Required commands

The project must support:

```bash
uv sync
uv run prepare.py
uv run train.py
uv run pytest -q
uv run python scripts/eda.py
uv run python scripts/leaderboard.py
```

After the AutoResearch harness is implemented:

```bash
uv run python scripts/run_experiment.py
```

Final holdout evaluation must be a separate human-run command:

```bash
uv run python scripts/final_eval.py --artifact artifacts/approved/<artifact-name>
```

---

# 5. Configuration

## `configs/benchmark.yaml`

Create explicit benchmark configuration rather than scattering constants throughout code.

Recommended defaults:

```yaml
seed: 42
split:
  train: 0.70
  validation: 0.15
  hidden_test: 0.15

model_constraints:
  min_validation_roc_auc: 0.70
  max_brier_score: 0.25

policy_constraints:
  max_human_review_rate: 0.35
  max_treatment_cost_per_account: 250.0
  require_full_coverage: true

experiment:
  wall_clock_seconds: 180
  max_experiments: 50
  patience: 12
```

The exact values are project assumptions and must be documented in README.

## `configs/business_simulation.yaml`

Create versioned simulation economics:

```yaml
currency: INR

costs:
  NO_CONTACT: 0.0
  DIGITAL_REMINDER: 5.0
  PAYMENT_PLAN_REVIEW: 50.0
  HUMAN_REVIEW: 150.0

experience_penalties:
  unnecessary_digital_contact: 5.0
  unnecessary_payment_plan_review: 25.0
  unnecessary_human_review: 100.0

missed_opportunity:
  no_contact_default_rate: 0.015

recovery_rate_bounds:
  DIGITAL_REMINDER: [0.02, 0.10]
  PAYMENT_PLAN_REVIEW: [0.06, 0.20]
  HUMAN_REVIEW: [0.08, 0.25]

exposure:
  max_fraction_of_limit: 1.0
  max_exposure_inr: 250000.0
```

These are explicitly **simulated economics**, not claims about real collections operations.

---

# 6. Data Pipeline — Protected

## `autocollections/data/loader.py`

Responsibilities:

1. Fetch UCI dataset ID 350 using `ucimlrepo`.
2. Normalize column names to stable uppercase or snake_case names.
3. Normalize target to `default_next_month`.
4. Validate expected row count is 30,000 unless a test fixture/subset flag is used.
5. Persist a local Parquet cache to:

```text
data/processed/uci_credit_default.parquet
```

6. Store dataset metadata and retrieval timestamp in:

```text
data/processed/dataset_metadata.json
```

7. Record dataset source, UCI ID, license, and a SHA-256 hash of the processed data.

## `autocollections/data/schema.py`

Define canonical feature groups.

### Allowed model features

Use only financial/behavioral variables for the official benchmark:

```text
LIMIT_BAL
PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6
BILL_AMT1
BILL_AMT2
BILL_AMT3
BILL_AMT4
BILL_AMT5
BILL_AMT6
PAY_AMT1
PAY_AMT2
PAY_AMT3
PAY_AMT4
PAY_AMT5
PAY_AMT6
```

### Excluded model features

Do not permit these in the official model/policy feature matrix:

```text
SEX
EDUCATION
MARRIAGE
AGE
```

If dataset column naming differs, normalize accordingly and write tests.

## `autocollections/data/splits.py`

Create one deterministic split manifest using seed 42.

Requirements:

- Stratify by target.
- Train 70%.
- Validation 15%.
- Hidden test 15%.
- Persist only row indices / opaque sample IDs in `split_manifest.json`.
- Never split differently across experiments.
- Include SHA-256 hash of the split manifest.

The hidden test must not be evaluated by the autonomous experiment loop.

---

# 7. Approved Feature Engineering

## `autocollections/features/approved_features.py`

This module provides **protected helper functions** that the agent may call from `train.py`; the agent cannot modify this module during research.

Implement reusable deterministic features:

### Raw financial features

- credit limit;
- repayment-status months;
- bill amounts;
- payment amounts.

### Derived features

At minimum:

```text
latest_utilization = max(BILL_AMT1, 0) / max(LIMIT_BAL, 1)
mean_utilization_6m
max_utilization_6m
payment_to_bill_ratio_1m
mean_payment_to_bill_ratio_6m
repayment_delay_mean
repayment_delay_max
repayment_delay_recent
repayment_delay_trend
bill_trend
payment_trend
months_with_positive_delay
months_with_payment
```

Use clipping/epsilon logic to handle zero/negative bills safely.

**Important:** `train.py` may choose which approved features to use and may create additional transformations from allowed financial columns, but it may never add excluded demographic columns.

---

# 8. EDA

## `scripts/eda.py`

Create a reproducible, non-notebook EDA script.

Output:

```text
reports/eda_summary.json
reports/figures/target_distribution.png
reports/figures/repayment_status_distribution.png
reports/figures/utilization_distribution.png
reports/figures/payment_bill_relationship.png
```

EDA must report:

- row count;
- feature count;
- target prevalence;
- missing values;
- duplicate rows;
- numeric ranges;
- suspicious categorical values;
- target distribution;
- summary of excluded demographic columns;
- financial feature distributions.

Do not “clean away” unusual values without documenting the decision.

---

# 9. Baseline Risk Models

Before AutoResearch, implement protected baseline reporting in:

```text
autocollections/evaluation/baselines.py
```

Required baselines:

### B0 — majority/default-rate reference

A non-learned reference.

### B1 — logistic regression on raw approved features

Pipeline:

```text
median imputation -> StandardScaler -> LogisticRegression
```

### B2 — logistic regression + approved derived features

### B3 — calibrated logistic regression

Use `CalibratedClassifierCV` with a fixed method (`sigmoid` initially).

### B4 — optional tree baseline

Only after B0-B3 pass. Prefer HistGradientBoostingClassifier from scikit-learn before adding external dependencies.

Report:

```text
ROC-AUC
PR-AUC
F1 at diagnostic threshold 0.5
Brier score
log loss
```

These are diagnostics, not the primary business score.

---

# 10. Treatment Policy Contract

The research candidate in `train.py` must expose a narrow interface.

Create a protected protocol or validation contract similar to:

```python
class CandidatePolicyProtocol(Protocol):
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    def predict_default_probability(self, X: pd.DataFrame) -> np.ndarray: ...

    def choose_actions(self, X: pd.DataFrame) -> np.ndarray: ...

    def export(self, path: Path) -> dict: ...
```

`choose_actions()` must return exactly one of:

```text
0, 1, 2, 3
```

for every row.

No abstention in the MVP. Uncertain cases should map to `HUMAN_REVIEW`.

---

# 11. Business Simulator — Protected

## Important modeling limitation

The UCI dataset contains default outcomes but **not counterfactual treatment outcomes**. Therefore Phase 1 must use a deterministic, seeded simulation to compare policies. The simulator is a benchmark tool, not a causal model.

## `autocollections/simulator/business_simulator.py`

The simulator receives:

```text
X_eval
observed default target y_eval
candidate actions
predicted default probabilities
```

The simulator uses `y_eval` only inside the protected evaluator.

### Exposure proxy

Define:

```python
exposure = clip(
    max(BILL_AMT1, 0),
    0,
    min(LIMIT_BAL, MAX_EXPOSURE_INR),
)
```

If `BILL_AMT1 <= 0`, optionally fall back to mean positive six-month bill amount, capped by limit.

### Hidden treatment-response proxy

Create two deterministic profile scores from financial behavior only:

```text
ability_proxy
engagement_proxy
```

Examples:

- `ability_proxy`: based on recent payment-to-bill ratios, payment consistency, and utilization;
- `engagement_proxy`: based on existence/recency of payments and improving repayment delays.

Scale each to `[0, 1]` using fixed protected transformations.

Do not expose simulator-internal row-level response values to `train.py`.

### Recovery simulation when `y == 1`

For defaulted evaluation examples, estimate recoverable value as:

```text
recovery = exposure * action_recovery_rate(profile)
```

with action-specific rates bounded by config.

A reasonable deterministic interpolation:

```text
DIGITAL_REMINDER:
    low recovery rate; benefits most from high engagement

PAYMENT_PLAN_REVIEW:
    medium recovery rate; benefits from moderate/high engagement and constrained ability

HUMAN_REVIEW:
    highest potential rate; benefits for high exposure / difficult cases, but costs more
```

Do not use randomness in official evaluation unless the seed is fixed and reproducibility tests prove identical outputs.

### Non-default (`y == 0`) cases

Set recovery benefit to zero. Any treatment other than `NO_CONTACT` incurs:

- treatment cost;
- unnecessary-contact/customer-experience penalty.

### Missed-opportunity penalty

For `y == 1` and `NO_CONTACT`, apply a small configured penalty proportional to exposure to represent missed opportunity.

### Utility per account

```text
utility = simulated_recovery
          - treatment_cost
          - unnecessary_contact_penalty
          - missed_opportunity_penalty
          - policy_penalty
```

Aggregate:

```text
total_simulated_recovery
total_treatment_cost
total_experience_penalty
total_missed_opportunity_penalty
total_business_utility
business_utility_per_account
```

The simulator must have extensive unit tests with hand-calculated tiny fixtures.

---

# 12. Protected Evaluation Metrics

## `autocollections/evaluation/metrics.py`

Return both ML and business metrics.

### Required ML diagnostics

```text
roc_auc
pr_auc
brier_score
log_loss
```

### Required policy/business metrics

```text
business_utility_total
business_utility_per_account
simulated_recovery_total
treatment_cost_total
experience_penalty_total
missed_opportunity_penalty_total
human_review_rate
digital_reminder_rate
payment_plan_review_rate
no_contact_rate
coverage
```

### Constraint metrics

```text
invalid_action_count
policy_violation_count
model_feature_violation_count
runtime_seconds
```

---

# 13. Primary Objective — Protected

## `autocollections/evaluation/objective.py`

Use **constrained business-utility maximization**.

A candidate is invalid if any hard constraint fails.

Hard constraints:

```text
coverage == 1.0
invalid_action_count == 0
policy_violation_count == 0
no excluded demographic feature usage
human_review_rate <= configured max
runtime <= experiment budget
```

Quality guardrails:

```text
roc_auc >= min_validation_roc_auc
brier_score <= max_brier_score
```

Primary score:

```python
if hard_constraint_failed:
    primary_score = -1e18
else:
    primary_score = business_utility_per_account
```

Do not combine every metric into a fragile weighted sum. Keep the main design easy to explain:

> Maximize business utility **subject to** model-quality and policy constraints.

Also report lift relative to fixed baselines:

```text
utility_lift_vs_always_no_contact_pct
utility_lift_vs_always_digital_pct
utility_lift_vs_rule_policy_pct
```

---

# 14. Baseline Treatment Policies

Implement protected baseline policies in `baselines.py`.

Required:

### P0 — Always no contact

```text
all -> NO_CONTACT
```

### P1 — Always digital

```text
all -> DIGITAL_REMINDER
```

### P2 — Static rule baseline

Start with:

```text
p(default) < 0.20   -> NO_CONTACT
0.20 - 0.45         -> DIGITAL_REMINDER
0.45 - 0.70         -> PAYMENT_PLAN_REVIEW
>= 0.70             -> HUMAN_REVIEW
```

Use the calibrated logistic baseline probabilities.

These thresholds are intentionally simple and may later be improved by AutoResearch.

---

# 15. `prepare.py` — Protected Entry Point

`prepare.py` should mirror the conceptual role of Karpathy's AutoResearch `prepare.py`.

Responsibilities:

1. Fetch/cache data.
2. Validate data.
3. Build deterministic split manifest.
4. Materialize approved feature views.
5. Compute and cache protected baselines.
6. Provide runtime utilities imported by `train.py`.
7. Expose a narrow validation evaluator.
8. Never expose hidden-test metrics in the normal experiment path.

`uv run prepare.py` should produce a concise summary such as:

```text
AutoCollections prepare
-----------------------
Dataset: UCI 350
Rows: 30000
Target prevalence: 0.xxxx
Train: 21000
Validation: 4500
Hidden test: 4500
Allowed raw features: 19
Excluded demographic features: 4
Dataset hash: <sha256>
Split hash: <sha256>
Baselines written: reports/baselines.json
STATUS: READY
```

---

# 16. `train.py` — The Only Agent-Editable Research File

This is the most important file.

The initial version should implement a clean, understandable candidate:

1. select approved raw + derived features;
2. median imputation;
3. standard scaling;
4. logistic regression;
5. probability calibration;
6. simple action thresholds;
7. export candidate artifact;
8. call protected validation evaluator;
9. print/write machine-readable experiment summary.

## What the research agent may change in `train.py`

Allowed:

- approved feature selection;
- new transformations derived from approved financial/behavioral columns;
- model family using already-approved dependencies;
- hyperparameters;
- class weighting;
- calibration method;
- treatment thresholds;
- decision policy using prediction confidence and approved features;
- simple ensembles if within budget;
- artifact metadata for its own candidate.

Forbidden:

- importing excluded demographic features into the candidate model/policy;
- reading hidden-test labels/files;
- modifying protected evaluator/simulator/constraints;
- changing business simulation economics;
- modifying tests;
- network calls;
- installing packages during an experiment;
- changing `program.md` during autonomous research;
- calling `scripts/final_eval.py`.

## Experiment output

Every `uv run train.py` must create:

```text
reports/experiments/latest_summary.json
artifacts/candidates/latest/
```

and print a stable block:

```text
--- AUTOCOLLECTIONS_RESULT ---
primary_score: <float>
roc_auc: <float>
pr_auc: <float>
brier_score: <float>
business_utility_per_account: <float>
business_utility_total: <float>
simulated_recovery_total: <float>
treatment_cost_total: <float>
human_review_rate: <float>
coverage: <float>
constraint_status: PASS|FAIL
runtime_seconds: <float>
artifact_sha256: <hash>
--- END_RESULT ---
```

Do not print hidden-test results.

---

# 17. Candidate Artifact

Export candidate policy to:

```text
artifacts/candidates/<timestamp-or-run-id>/
```

Contents:

```text
model.joblib
feature_schema.json
policy_config.json
manifest.json
```

`manifest.json` must include:

```text
artifact_version
created_at
source_git_commit
dataset_hash
split_hash
feature_names
excluded_feature_check
model_class
model_params
calibration_method
policy_description
validation_metrics
python_version
dependency_lock_hash
```

Hash the artifact contents.

---

# 18. Protected File Integrity

## `scripts/check_protected.py`

Create a manifest of protected files and SHA-256 hashes.

Protected during research:

```text
prepare.py
configs/benchmark.yaml
configs/business_simulation.yaml
autocollections/data/**
autocollections/features/**
autocollections/simulator/**
autocollections/evaluation/**
autocollections/artifacts/**
tests/**
scripts/final_eval.py
program.md
```

Writable/editable by the research agent:

```text
train.py
results.tsv
reports/experiments/**
artifacts/candidates/**
run.log
```

`run_experiment.py` must check protected hashes before and after each experiment. If any protected file changes, mark the experiment `INVALID`, restore protected files from Git, and reject the candidate.

This is an MVP integrity layer. Document that stronger OS/container isolation is a later hardening step.

---

# 19. `program.md` — AutoResearch Research Charter

Create `program.md` with content equivalent to the following.

```markdown
# AutoCollections Research Mission

Improve the customer-treatment candidate in `train.py` to maximize the official protected validation `primary_score` while every constraint passes.

## Editable scope

You may modify **only `train.py`**.

Do not modify:
- `prepare.py`
- `program.md`
- `configs/`
- `autocollections/data/`
- `autocollections/features/`
- `autocollections/simulator/`
- `autocollections/evaluation/`
- `tests/`
- `scripts/final_eval.py`
- hidden-test data or split manifests

Do not add excluded demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) to model or policy features.

## Research protocol

For every experiment:

1. Read the current `train.py` and prior `results.tsv`.
2. State exactly one falsifiable hypothesis.
3. Prefer the smallest coherent code change that tests that hypothesis.
4. Run the official experiment command:
   `uv run python scripts/run_experiment.py --candidate-only`
5. Read only the official aggregate validation summary.
6. A change is KEEP only if:
   - primary score improves over the current best;
   - all constraints pass;
   - protected-file integrity passes;
   - the run finishes within budget.
7. If not, revert `train.py` to the previous accepted state.
8. Append exactly one row to `results.tsv`.
9. Commit only accepted changes.
10. Continue until experiment limit/patience is reached.

## Research strategy

Start simple.

High-value areas:
- derived financial behavior features;
- feature selection;
- class weighting;
- logistic-regression regularization;
- calibration;
- tree baselines using approved installed packages;
- action thresholds;
- confidence-aware human-review fallback;
- cost-sensitive policy decisions.

Avoid:
- architecture complexity without measured gain;
- duplicate experiments;
- tuning against hidden test;
- changing the benchmark or simulator;
- large dependency additions.

For surprising large gains, rerun with the same official protocol and verify reproducibility before committing.

## Stop conditions

Stop if any of these occurs:
- maximum experiment count reached;
- patience exhausted;
- repeated invalid/crashing experiments;
- no valid improvement path remains.
```

---

# 20. `results.tsv`

Create the file with header:

```text
experiment_id	git_commit	status	primary_score	roc_auc	pr_auc	brier_score	utility_per_account	recovery_total	treatment_cost	human_review_rate	runtime_seconds	hypothesis	description	artifact_hash
```

Possible statuses:

```text
BASELINE
KEEP
DISCARD
INVALID
CRASH
TIMEOUT
```

Append one row per experiment. Never silently overwrite history.

---

# 21. Experiment Runner

## `scripts/run_experiment.py`

This is protected infrastructure and must not contain model logic.

Responsibilities:

1. check working tree state;
2. capture current best commit;
3. verify protected hashes;
4. generate experiment ID;
5. run `uv run train.py` with wall-clock timeout;
6. capture stdout/stderr to `run.log` and per-experiment log;
7. parse `latest_summary.json` rather than fragile console text;
8. re-check protected hashes;
9. compare candidate primary score to current best accepted score;
10. validate constraints;
11. output decision suggestion `KEEP` / `DISCARD` / `INVALID` / `CRASH` / `TIMEOUT`;
12. append structured run metadata;
13. **do not automatically commit unless called in autonomous mode.**

Support modes:

```bash
uv run python scripts/run_experiment.py --candidate-only
uv run python scripts/run_experiment.py --auto-decision
```

`--candidate-only` evaluates but leaves Git decision to the research agent.

`--auto-decision` may commit/revert after all tests and checks pass.

Before using `--auto-decision`, unit tests must pass.

---

# 22. Git Keep/Revert Behavior

For an accepted candidate:

```bash
git add train.py results.tsv
git commit -m "research: <short experiment description>"
```

Do not commit generated raw data or temporary logs.

For a rejected candidate:

```bash
git restore --source=<current-best-commit> -- train.py
```

Do not use destructive repository-wide reset if it could remove user work unrelated to `train.py`.

The autonomous loop must be conservative with Git.

---

# 23. Final Hidden-Test Evaluation

## `scripts/final_eval.py`

This command must not be used during AutoResearch.

Process:

1. Require clean Git state.
2. Require an explicit candidate artifact path.
3. Verify artifact manifest and source commit.
4. Freeze candidate.
5. Evaluate exactly once on hidden test.
6. Produce:

```text
reports/final_hidden_test.json
reports/final_report.md
```

Report:

```text
hidden ROC-AUC
hidden PR-AUC
hidden Brier score
hidden business utility/account
hidden total simulated recovery
hidden treatment cost
hidden human review rate
baseline comparisons
constraint status
```

Also include a mandatory disclaimer:

> Business utility and recovery values are outputs of a reproducible simulator built on a public default-risk dataset. They are not observed intervention outcomes and must not be interpreted as real bank savings or causal treatment effects.

Do not allow the research agent to rerun hidden test repeatedly.

---

# 24. Leaderboard

## `scripts/leaderboard.py`

Read `results.tsv` and print:

```text
Top accepted experiments by primary score
Current best commit
Score progression
Accepted vs rejected counts
Best ROC-AUC
Lowest Brier score
Best business utility/account
Human-review rate
```

Also generate:

```text
reports/figures/research_progress.png
```

Plot experiment number vs primary score, marking KEEP and DISCARD.

---

# 25. Tests

Tests are part of the evaluator and must be protected from the research agent.

## `test_data.py`

Verify:

- dataset loads;
- 30,000 rows in full mode;
- target binary;
- deterministic processed hash for unchanged inputs;
- no target included in X;
- no missing target values.

## `test_features.py`

Verify:

- excluded demographic features never appear in official feature matrix;
- derived features finite;
- no division-by-zero output;
- same input gives same feature output.

## `test_simulator.py`

Use a tiny hand-built fixture to verify exact:

- exposure;
- action cost;
- recovery;
- penalty;
- total utility.

Check simulator is deterministic.

## `test_objective.py`

Verify:

- better utility wins if constraints pass;
- failing hard constraint produces disqualifying score;
- coverage < 1 fails;
- invalid action fails;
- excess human-review rate fails;
- low ROC-AUC fails.

## `test_baselines.py`

Use tiny fixtures where always-no-contact, always-digital, and static rules can be manually verified.

## `test_train_contract.py`

Run initial candidate on small fixture and verify:

- probabilities in `[0, 1]`;
- exactly one action per input row;
- actions in `{0,1,2,3}`;
- artifact exports.

## `test_protected_integrity.py`

Simulate protected-file modification and verify guard rejects experiment.

## `test_reproducibility.py`

Two identical runs with same seed/config should produce matching aggregate metrics within a strict tolerance.

---

# 26. CI

Add `.github/workflows/ci.yml` only after local tests pass.

Run on push / pull request:

```text
uv sync --locked
ruff check .
pytest -q
```

Do not download the full UCI dataset during every unit-test run. Use small deterministic fixtures for CI tests. A separate optional integration test may fetch UCI data.

---

# 27. README Requirements

`README.md` must explain:

1. business problem;
2. why the project uses AutoResearch;
3. how it maps to Karpathy's `prepare.py` / `train.py` / `program.md` pattern;
4. dataset source and license;
5. that Phase 1 treatment outcomes are simulated;
6. excluded demographic features;
7. system architecture;
8. quick start;
9. baseline metrics after they are actually produced;
10. AutoResearch loop;
11. results ledger;
12. final hidden-test discipline;
13. limitations;
14. roadmap.

Never place placeholder performance numbers in README as if measured.

---

# 28. Phase-by-Phase Implementation Plan

Codex must implement in this order.

## Phase 1 — Repository + environment

Tasks:

- create structure;
- configure `pyproject.toml`;
- create `.gitignore`;
- verify `uv sync`;
- add empty/reference README sections.

Acceptance:

```bash
uv sync
uv run pytest -q
```

works with initial tests/fixtures.

---

## Phase 2 — Data + EDA

Tasks:

- implement UCI loader;
- normalize schema;
- exclude demographics from official model view;
- create deterministic split manifest;
- implement `prepare.py` dataset stage;
- create EDA script.

Acceptance:

```bash
uv run prepare.py
uv run python scripts/eda.py
```

succeeds and writes hashes/reports.

---

## Phase 3 — Baseline risk model

Tasks:

- logistic regression baseline;
- derived features;
- calibrated logistic baseline;
- baseline metrics report.

Acceptance:

- ROC-AUC, PR-AUC, Brier score and log loss produced;
- reproducibility tests pass;
- no demographic model features.

---

## Phase 4 — Business simulator + baseline treatment policies

Tasks:

- implement exposure;
- profile proxies;
- treatment recovery simulator;
- costs and penalties;
- business utility;
- P0/P1/P2 treatment baselines.

Acceptance:

- tiny hand-calculated simulator tests pass;
- baseline business report generated;
- simulation assumptions documented.

---

## Phase 5 — Protected evaluator + initial `train.py`

Tasks:

- candidate policy contract;
- objective/constraints;
- artifact export;
- stable JSON summary;
- initial agent-editable `train.py`.

Acceptance:

```bash
uv run train.py
```

produces complete validation summary, valid artifact, and PASS/FAIL constraint status.

---

## Phase 6 — AutoResearch loop

Tasks:

- `program.md`;
- `results.tsv`;
- protected-hash manifest;
- experiment runner;
- safe Git keep/revert;
- leaderboard.

Acceptance:

Run at least 10 unattended/semiautonomous experiments correctly, including:

- at least one KEEP;
- at least one DISCARD;
- one intentionally invalid protected-file modification rejected by test/harness;
- one timeout/crash path handled without corrupting the repository.

Then expand to 20-50 experiments.

---

## Phase 7 — Final evaluation + lightweight product layer

Only start after Phase 6 is trustworthy.

Core finalization:

- freeze best candidate;
- run one hidden-test evaluation;
- generate final report;
- produce research-progress plot;
- update README with actual metrics.

### Optional lightweight serving layer

After core research is complete, add:

```text
FastAPI + Pydantic
Streamlit dashboard
Docker
```

Keep this separate from the research harness.

FastAPI MVP endpoints:

```text
POST /v1/decision
GET /v1/policy
GET /health
```

The API should return a simulated next-best-action decision only. It must not send messages, initiate collections actions, or connect to a bank.

Dashboard:

```text
best business utility
experiment count
KEEP/DISCARD distribution
risk-score distribution
action distribution
human-review rate
simulated recovery
simulated treatment cost
ML diagnostics
```

---

## Phase 8 — Optional GenAI / DebtBench extension

Do not implement this until the classical AutoResearch project is complete.

Future direction:

- persona-aware conversation benchmark;
- LLM-based treatment/dialogue strategy;
- quality, affordability, interaction and negotiation metrics;
- strict safety/human-review controls.

This phase should be a separate branch or module, not mixed into the MVP.

---

# 29. Recommended Initial `train.py` Candidate

Start with an intentionally strong-but-simple baseline:

```text
Approved raw + derived financial features
        ↓
Median imputation
        ↓
StandardScaler
        ↓
LogisticRegression(class_weight="balanced")
        ↓
CalibratedClassifierCV(method="sigmoid")
        ↓
P(default)
        ↓
Static thresholds
        ↓
NO_CONTACT / DIGITAL_REMINDER / PAYMENT_PLAN_REVIEW / HUMAN_REVIEW
        ↓
Protected evaluation
```

Do not start with neural networks.

---

# 30. Suggested AutoResearch Experiment Ideas

These are ideas, not mandatory hard-coded experiments. The agent should propose and test them one at a time.

### Features

- latest utilization;
- mean utilization;
- repayment delay trend;
- payment/bill ratios;
- payment consistency;
- recent-vs-old behavior weighting;
- capped/log-transformed bill amounts;
- interaction between utilization and recent delay.

### Model

- logistic `C` regularization;
- balanced/unbalanced weights;
- HistGradientBoosting;
- calibrated tree model;
- simple ensemble if justified.

### Calibration

- sigmoid vs isotonic;
- confidence thresholds;
- calibration-aware human-review fallback.

### Policy

- optimize treatment thresholds;
- use exposure-aware escalation;
- reduce human review for low-exposure uncertain cases;
- route high-confidence moderate-risk cases digitally;
- use probability + utilization/ability proxy for payment-plan review.

### Complexity discipline

If a more complex model improves primary score by a negligible amount, prefer the simpler policy and document the trade-off.

---

# 31. What Counts as Success

The project is successful when all of the following are true:

1. The UCI dataset pipeline is reproducible.
2. Demographic columns are excluded from official model/policy features.
3. Classical risk baselines are measured.
4. Business simulator is deterministic and unit tested.
5. Business utility is clearly labeled simulated.
6. `prepare.py` is protected infrastructure.
7. `train.py` is the only agent-editable research implementation.
8. `program.md` controls the research loop.
9. `results.tsv` records every experiment.
10. Failed changes are safely reverted.
11. Accepted changes are committed with clear history.
12. The best candidate beats a predefined simple treatment-policy baseline on validation.
13. A frozen candidate is evaluated once on hidden test.
14. Final report includes negative results and limitations.
15. No real-world savings or causal-treatment claims are fabricated.

---

# 32. Resume Metrics — Only Fill After Real Experiments

Do **not** use these as claims until the system generates real measured values.

Possible final bullets:

- Built an **AutoResearch-based customer-treatment optimizer** that autonomously proposed, evaluated, committed, and reverted risk-model and next-best-action strategies under protected business-utility and policy constraints.
- Developed a reproducible credit-risk benchmark over **30,000 public customer records**, with calibrated default probabilities, deterministic treatment simulation, hidden-test isolation, and cost-sensitive policy evaluation.
- Ran **[N] autonomous experiments**, accepting **[K]** improvements and increasing simulated business utility by **[X]%** versus the predefined rule baseline while maintaining **[Y] ROC-AUC**, **[Z] Brier score**, and a **[H]% human-review rate**.

If Phase 7 is implemented:

- Productionized the approved policy artifact using FastAPI, versioned manifests, rollback-safe model loading, and a dashboard for simulated recovery, treatment cost, model calibration, action mix, and experiment lineage.

---

# 33. Topics the Developer Should Understand

Study these while implementing:

## AutoResearch

- Karpathy AutoResearch design
- `prepare.py` vs `train.py`
- `program.md`
- fixed-budget experiments
- Git keep/revert
- experiment lineage
- validation overfitting
- hidden-test discipline
- metric gaming

## Machine Learning

- logistic regression
- class imbalance
- ROC-AUC vs PR-AUC
- probability calibration
- Brier score
- log loss
- feature scaling
- tree ensembles
- cost-sensitive learning
- threshold optimization

## Financial/Business Analytics

- probability of default
- delinquency / repayment behavior
- customer treatment strategies
- next-best-action
- operational cost
- recovery/cure concepts
- human escalation
- business utility
- why treatment effects require causal/intervention data

## Experiment Design

- train/validation/hidden test
- stratification
- leakage
- reproducibility
- deterministic seeds
- baselines
- ablations
- confidence intervals
- multiple-testing/validation overfitting

## Responsible AI

- sensitive/protected attributes
- proxy discrimination
- human oversight
- auditability
- data minimization
- simulated vs real-world claims

## Software Engineering

- Python packaging with `uv`
- typed interfaces
- config-driven design
- hashing/artifact manifests
- unit/integration tests
- Git safety
- CI
- optional FastAPI/Streamlit/Docker

---

# 34. Explicit Out-of-Scope Items for MVP

Do not implement these until core acceptance criteria are met:

- real bank integration;
- real collections outreach;
- payment processing;
- SMS/WhatsApp/email sending;
- CRM integration;
- customer PII;
- automated legal/financial advice;
- LLM debt negotiation;
- autonomous customer messaging;
- reinforcement learning on real customers;
- online learning from production outcomes;
- Kubernetes;
- Kafka;
- multi-agent swarms;
- vector databases.

---

# 35. Source References

Use these as conceptual/data sources in README and documentation:

1. Andrej Karpathy, `autoresearch` GitHub repository — core three-file AutoResearch design and fixed-budget keep/discard loop.
   - https://github.com/karpathy/autoresearch

2. UCI Machine Learning Repository — Default of Credit Card Clients, dataset ID 350.
   - https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
   - DOI: 10.24432/C55S3H
   - License: CC BY 4.0

Do not imply that Karpathy or UCI authors endorse this project or its business-simulation design.

---

# 36. Codex Working Instructions

When implementing this specification:

1. Inspect the existing repository first; preserve any user-created content unless it conflicts with this specification.
2. Implement one phase at a time.
3. Do not skip tests to move faster.
4. Do not introduce extra infrastructure without necessity.
5. Keep public interfaces typed and small.
6. Prefer deterministic behavior.
7. After every phase:
   - run tests;
   - run the relevant command;
   - summarize files created/changed;
   - report any assumptions;
   - stop if a foundational test fails.
8. Do not fabricate metrics. Only report values actually produced by code.
9. Do not run final hidden-test evaluation during iterative research.
10. Before autonomous research begins, make sure a human can manually run one candidate end-to-end and understand every metric.

## First action Codex should take

Start with **Phase 1 and Phase 2 only**:

- inspect repository;
- create the minimal structure;
- configure `uv`;
- implement UCI loader/schema/splits;
- implement `prepare.py` dataset preparation;
- implement EDA;
- add tests;
- run them;
- report the actual outputs.

Do **not** start AutoResearch agent loops until the dataset, baseline evaluation, simulator, and protected objective are all validated.

---

# Final Design Summary

```text
                 PUBLIC UCI CREDIT DATA
                           │
                           ▼
                    prepare.py
                     PROTECTED
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   deterministic       approved        protected
      splits            features        evaluator
                                             │
                                             │ aggregate metrics
                                             ▼
 program.md  ───► AI Research Agent ───► train.py
                 (research process)       EDITABLE
                                             │
                                             ▼
                                       candidate policy
                                             │
                                             ▼
                                      protected evaluator
                                             │
                                       score + constraints
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
                        KEEP                                  DISCARD
                          │                                     │
                       commit                                revert
                          └──────────────────┬──────────────────┘
                                             ▼
                                         results.tsv
                                             │
                                      next hypothesis
                                             │
                                             └──────────────► loop

                  AFTER RESEARCH IS FROZEN
                                             │
                                             ▼
                                     hidden-test evaluation
                                             │
                                             ▼
                                    approved policy artifact
```

The core philosophy is simple:

> **The agent is allowed to search for a better treatment policy. It is not allowed to redefine the benchmark, inspect the final answer key, or claim simulated economics as real business impact. Evidence decides which experiment survives; a human decides whether the artifact is fit for any demonstration or deployment.**
