---
id: refactor-evaluate-size-splits
title: Split four modules near the size cap without changing behaviour
status: ready
depends_on: --
blocks: generate-no-mapping-ablation, evaluate-prior-shift-comparator, evaluate-added-row-contradictions, gaps-is-a-conditional-fragments, ontology-diabetes-diagnosis-codes, privacy-donor-attacks, evaluate-trace-attribution, gaps-share-based-quota, ontology-per-dataset-releases
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Split four modules near the size cap without changing behaviour

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Four files that every later unit of this set extends sit near the line cap that `make sizes`
enforces: `src/bko/evaluate/named_runs.py`, `src/bko/generate/direct.py`,
`src/bko/evaluate/membership.py` and `src/bko/evaluate/run.py`. A split that lands alone and first
keeps each later unit's diff about its own change.

## Scope

**In.** No behaviour change.

- `named_runs.py`: `NamedRun` to `named_run.py`; the Diabetes families and `_slices` to
  `runs_diabetes.py`.
- `generate/direct.py`: `FieldGroup`, `field_groups` and `free_fields` to `generate/fields.py`;
  `_pick`, `label_group`, `_observe_group`, `_label_groups` and `_records` to
  `generate/donors.py`.
- `membership.py`: `REFERENCE` and `reference_of` to `evaluate/reference.py`.
- `run.py`: `_replica_scores`, `_require_regression`, `_suffixed`, `_rows` and `_scoped` to
  `evaluate/run_scores.py`.
- Each old module re-exports the moved names, so no import elsewhere changes.
- Each ADR whose Enforcement paths name moved code is updated to the new path.

**Out.**

- `src/bko/evaluate/holdout_candidates.py`, `src/bko/generate/flight.py` and
  `src/bko/evaluate/experiment.py`.
- Any signature change.

## Acceptance

- `DRAW_DIGEST` (`tests/test_generate_direct.py`) and `ADDED_DIGEST`
  (`tests/test_evaluate_direct.py`) are unchanged.
- No test file changes.
- `make sizes` passes with each of the four files at least 100 lines under its cap.
- `make check` passes.
