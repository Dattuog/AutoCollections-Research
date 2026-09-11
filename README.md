## Live Demo

Try the public demo:

https://autocollections-research.streamlit.app/

The demo uses synthetic inputs only and the frozen exp_039 model.
It does not retrain the model or access hidden-test data.


# AutoCollections Research

AutoCollections Research applies the AutoResearch pattern to a public credit-default
dataset: protected preparation and evaluation code surrounds one later agent-editable
candidate. The eventual objective is simulated business utility under model-quality,
cost, operational, and policy constraints—not classification accuracy alone.

## Current scope

Phases 1–4 provide the reproducible environment, UCI data loader, approved financial
feature view, deterministic train/validation/hidden-test split, scripted EDA, and
default-risk and simulated treatment-policy baselines. Candidate training, experiment
automation, and final evaluation belong to later phases and are not implemented yet.

## Data and responsible-use boundary

The project uses the UCI *Default of Credit Card Clients* dataset (ID 350), licensed
CC BY 4.0. `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are excluded from the official
model/policy feature matrix. This repository uses public data and must not make real
customer decisions or process real PII.

The dataset has no historical treatment outcomes. Any recovery, cost, or treatment
effect introduced in a later phase will be a deterministic simulation—not observed
bank savings or causal evidence.

## Architecture

- `prepare.py`: protected dataset preparation and split creation.
- `autocollections/data/`: protected loading, validation, schema, and split logic.
- `autocollections/features/`: protected approved feature view.
- `train.py`: reserved for the later agent-editable candidate.
- `program.md` and the evaluator/simulator: reserved for later protected phases.

## Quick start

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
uv run prepare.py
uv run python scripts/eda.py
uv run python -m autocollections.evaluation.baselines
uv run python scripts/run_policy_baselines.py
uv run train.py
```

If UCI retrieval is unavailable, place the original workbook at
`data/raw/default_of_credit_card_clients.xls`; raw and processed data are ignored by Git.

## Benchmark assumptions

The split seed is 42 with 70% train, 15% validation, and 15% hidden test. Later phases
must enforce the configured quality and policy constraints. These are project benchmark
assumptions, not production policy recommendations.

The source data contains 35 duplicate rows beyond the first copy when all 23 features
and the target are compared. Comparing features without the target finds 56 duplicates;
21 feature groups contain conflicting targets. The 19 approved model features produce
817 duplicates across 134 groups, 85 with conflicting targets. No rows are removed.
Instead, the deterministic stratified split keeps every identical approved-feature group
within one split, preventing model-visible duplicate leakage while preserving exact
21,000/4,500/4,500 sizes and target counts.

## Validation baselines

These are measured on the canonical validation split at the neutral `0.5` threshold;
the hidden test was not evaluated.

| Model | ROC-AUC | PR-AUC | Brier | Log loss | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.5000 | 0.2211 | 0.1722 | 0.5283 | 0.0000 | 0.0000 | 0.0000 |
| Logistic, raw | 0.7310 | 0.5286 | 0.1426 | 0.4598 | 0.7201 | 0.2663 | 0.3888 |
| Logistic, raw + derived | 0.7745 | 0.5502 | 0.1352 | 0.4325 | 0.6644 | 0.3859 | 0.4882 |
| Logistic, derived + balanced | 0.7738 | 0.5470 | 0.1892 | 0.5704 | 0.4400 | 0.6523 | 0.5255 |
| Logistic, balanced + sigmoid calibration | 0.7738 | 0.5471 | 0.1356 | 0.4335 | 0.6557 | 0.3789 | 0.4803 |

The selected Phase 3 reference is unweighted logistic regression with approved raw and
derived features: it has the lowest validation Brier score and log loss. Class weighting
increases recall at `0.5` but substantially worsens calibration before sigmoid correction.
The AutoResearch keep/discard loop remains unimplemented until its later phase.

## Phase 4 treatment simulation

The fixed probability bands are `<0.20` no contact, `0.20–0.45` digital reminder,
`0.45–0.70` payment-plan review, and `>=0.70` human escalation. Simulation version
`phase4_v1` uses nominal simulated INR-equivalent units from
`configs/business_simulation.yaml`: action costs of 0/5/50/150, action-specific bounded
recovery rates, penalties for unnecessary contact, a 1.5% missed-opportunity charge on
untreated default exposure, and a 35% human-escalation capacity limit. These are benchmark
assumptions only, not observed treatment effects or real bank economics.

Validation comparison:

| Policy | Recovery | Cost | Experience/over-treatment | Missed opportunity | Policy penalty | Net utility | Utility/account |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always no contact | 0.00 | 0.00 | 0.00 | 667,057.05 | 0.00 | -667,057.05 | -148.2349 |
| Always digital reminder | 2,531,544.32 | 22,500.00 | 17,525.00 | 0.00 | 0.00 | 2,491,519.32 | 553.6710 |
| Always payment-plan review | 6,208,213.01 | 225,000.00 | 87,625.00 | 0.00 | 0.00 | 5,895,588.01 | 1,310.1307 |
| Always human escalation | 9,185,717.34 | 675,000.00 | 350,500.00 | 0.00 | 731,250.00 | 7,428,967.34 | 1,650.8816 |
| Risk-based four-action | 3,968,127.62 | 56,245.00 | 12,405.00 | 236,724.51 | 0.00 | 3,662,753.11 | 813.9451 |

Always human escalation breaches the capacity constraint for 2,925 accounts and incurs
731,250 simulated penalty units. Among policies without violations, always payment-plan
review has the highest v1 utility. The four-action policy remains the Phase 4 baseline;
its underperformance is evidence that later policy optimization and sensitivity analysis
are needed, not grounds to rewrite the assumptions after observing validation results.

### Phase 4.5 simulator audit

The simulator passes direct counterfactual checks: non-default rows receive no recovery,
no contact costs zero, treatment costs increase with intensity, unnecessary treatment
cannot create positive benefit, policy selection has no realized-target input, and only
approved features enter the risk model and simulator.

Always payment-plan review exceeds the risk-based policy by 2,232,834.90 simulated units
(496.19/account): 2,240,085.39 additional recovery and 236,724.51 avoided missed-opportunity
penalty outweigh 168,755.00 additional treatment cost and 75,220.00 additional
over-treatment penalty. Neither policy incurs a policy penalty.

A fixed one-at-a-time sensitivity grid varies payment-plan cost and over-treatment
penalty, digital and human costs, missed-opportunity rate, and all recovery bounds across
low/base/high values. All 13 scenarios retain the same ranking:

```text
always human escalation > always payment-plan review > risk-based four-action
> always digital reminder > always no contact
```

Human escalation violates the capacity constraint; payment-plan review is the highest
non-violating policy in every scenario. This broad static-policy dominance is a warning,
not a forced ranking rule. The evaluator logic is deterministic and internally coherent,
but the v1 economics are not yet discriminative enough to freeze for AutoResearch.

### Phase 4.6 heterogeneous simulator

`phase4_v2` is a separate simulator; v1 code and audit artifacts remain unchanged. It
retains the accepted risk thresholds, costs, penalties, exposure cap, human capacity,
canonical split, and `logistic_derived` model. Recovery is instead calculated as:

```text
exposure × action base effect × customer/action suitability
```

Base effects are 0.10 digital, 0.16 payment-plan review, and 0.22 human escalation.
Suitability is deterministic and uses three protected behavioral proxies:

- ability: 50% payment/bill ratio, 25% payment frequency, 25% inverse utilization;
- engagement: 40% payment frequency, 35% recent payment/bill ratio, 25% improving delay;
- severity: 45% recent delay, 30% persistent delay, 15% utilization, 10% exposure fraction.

The validation oracle chooses no contact for 3,625 accounts, digital for 252, payment-plan
review for 0, and human escalation for 623. This proves heterogeneous treatment response,
but always-human remains the unconstrained static winner across all seven v2 sensitivity
scenarios and violates capacity for 2,925 accounts. The risk policy is the highest-utility
non-violating fixed policy in every scenario. At this stage the unconstrained dominance
failed the preliminary freeze diagnostic; Phase 4.7 subsequently made feasibility the
authoritative deployability boundary.

### Phase 4.7 feasibility and capacity audit

Human capacity is now a hard feasibility boundary: policies above 35% human escalation
remain visible in raw diagnostics but are ineligible for deployable ranking, primary-score
selection, and future KEEP decisions. Three diagnostic policies assign the highest-risk
35% of validation accounts to human escalation and send the remainder to no contact,
digital, or payment-plan review.

The top-35%-human-else-digital policy is the highest feasible policy in the base case and
all seven accepted v2 sensitivity scenarios. It scores 3,623,150.14 simulated units versus
1,436,422.55 for the risk-based policy. This triggers the capacity-saturation warning:
the evaluator rewards filling all available human capacity regardless of the more selective
four-band treatment match. Payment-plan review also remains oracle-dominated with zero
oracle selections. This capacity-saturating policy is accepted as B4, the legitimate
strong feasible benchmark for research rather than a simulator failure.

## Protected research boundary

Phase 5 freezes the accepted v2 simulator and hard-feasibility semantics as evaluator
`phase5_evaluator_v1`. `train.py` is the only future agent-editable research file. It
receives approved training features/labels plus approved validation features and stable
row IDs, then emits exactly `sample_id`, `predicted_probability`, and `action`.
Validation outcomes remain inside the protected evaluator.

The primary score is feasible simulated utility per account. Infeasible candidates receive
`-1e18` and cannot be eligible for KEEP regardless of raw utility. The benchmark to beat is
`top35_human_else_digital`: 3,623,150.1362 total utility, or 805.1444747/account. The
initial `train.py` dry run reproduces it exactly. Automated experimentation is not yet
implemented; `program.md` and `results.tsv` only define the future protocol and ledger.

## Hidden-test discipline

The split manifest is created once. Autonomous experiments must receive aggregate
validation metrics only; final hidden-test evaluation is a separate, one-time human-run
step after a candidate is frozen.

## Limitations and roadmap

The public dataset measures default, not intervention response. Demographic exclusion
does not by itself eliminate proxy or fairness risks. The roadmap follows the numbered
phases in the implementation contract; serving infrastructure and GenAI extensions stay
out of scope until the research harness is validated.

## References

- [Karpathy AutoResearch](https://github.com/karpathy/autoresearch)
- [UCI dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
  DOI `10.24432/C55S3H`
