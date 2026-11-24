---
id: evaluate-control-runs-registered
title: Register the control runs on E6's and E3's frames and their EVIDENCE entries
status: blocked
depends_on: baselines-label-bootstrap, evaluate-prior-shift-comparator
blocks: evaluate-diabetes-control-results, evaluate-synthea-direct-results, gaps-share-based-quota
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Register the control runs on E6's and E3's frames and their EVIDENCE entries

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Readings fixed after a run are not readings. This unit fixes the runs, their arms and the
contrasts the control reading rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md) reads before
anything runs.

## Scope

**In.**

- A new `src/bko/evaluate/runs_direct.py` holding two run definitions:
  - a family on E6's frame at both fractions, sliced as E6 is, with arms `none`, `real`,
    `METHOD_ARMS`, `bayes-net`, `bko-direct`, `bko-nomap`, `label-bootstrap` and `prior-shift`,
    and candidate `bko-direct`
    (ADR
    [`research/diabetes-subgroup-removal-design`](../docs/adr/research/diabetes-subgroup-removal-design.md));
  - a run built from `E3` by `replace`, with the same removal, holdouts and
    `decontaminate=False`, so `compare --beside` pairs it with `runs/e3-7/`, with the same arms
    plus `bko-logic`
    (ADR [`research/synthea-downstream-design`](../docs/adr/research/synthea-downstream-design.md)).
    The `replace` resets `arms`, `replica_arms`, `spend_ceiling` and `candidate`, so the run keeps
    none of E3's paid `BKO_ARMS` and makes no paid call.
- `REFERENCE` entries for both, in `bko.evaluate.reference`.
- Two new [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entries, status `not yet run`, whose
  Configuration fixes the contrasts of the control reading rule and its margin.
- The two run-name tests, `tests/test_evaluate_experiment.py` and
  `tests/test_named_runs_slices.py`, list the new names.

**Out.**

- Running anything.

## Acceptance

- A test on the Synthea fixture frame shows the run's gap request asks rows for the removed cell.
- A test shows the Diabetes family's slices cover its holdouts once.
- `make check` passes.
