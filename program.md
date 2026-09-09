# AutoCollections Research Mission

Improve the treatment candidate in `train.py` to increase protected simulated business
utility over `top35_human_else_digital` while every hard constraint passes.

## Editable scope

During autonomous research, edit only `train.py`. Do not modify `prepare.py`, `program.md`,
configs, protected evaluator/simulator/data/feature code, tests, manifests, or historical
reports. Do not read or evaluate hidden-test outcomes. Do not use `SEX`, `EDUCATION`,
`MARRIAGE`, or `AGE` as model or policy inputs.

## Experiment protocol

1. Read the current `train.py` and `results.tsv`.
2. State one falsifiable hypothesis and make one coherent, minimal change.
3. Run the official candidate evaluation command.
4. Read only the aggregate validation result.
5. KEEP only when `PRIMARY_SCORE_ELIGIBLE=true` and primary score improves over the
   current accepted score; hard constraints outrank raw utility.
6. Otherwise REVERT `train.py` to the prior accepted state.
7. Append exactly one experiment row to `results.tsv`.

The 35% human limit is a maximum, not a target. Candidates may use less capacity. The
protected oracle is an evaluation-only upper bound; its row-level actions are forbidden.
Prefer the simpler implementation when performance is equivalent.

## Benchmark hierarchy

- B0: always no contact
- B1: always digital reminder
- B2: always payment-plan review
- B3: original risk-based four-action policy
- B4: `top35_human_else_digital`, the strong feasible benchmark to beat

`always_human_escalation` is infeasible and diagnostic only.

