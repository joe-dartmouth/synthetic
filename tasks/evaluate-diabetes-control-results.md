---
id: evaluate-diabetes-control-results
title: Run the Diabetes 130 control family and read each contrast's branch
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration, evaluate-quota-sweep-results, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the Diabetes 130 control family and read each contrast's branch

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The controls decide what the study can claim for `bko-direct`, and every control arm is free.

## Scope

**In.**

- `score` per slice, then `merge`, `rows`, `panel`, `privacy` and `compare`, as
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives them, for the Diabetes family
  [E12](../docs/EVIDENCE.md#e12-control-arms-on-the-diabetes-130-removal) names.
- The EVIDENCE entry's Result and Read against, naming the branch of the control reading rule in
  [`docs/RESEARCH.md`](../docs/RESEARCH.md) that each contrast falls in.

**Out.**

- Any change to an arm or a reading.

**Findings from planning.**

- At full removal `bko-direct` and `bko-nomap` draw one distribution on E6's frame (ADR
  [`research/no-mapping-ablation`](../docs/adr/research/no-mapping-ablation.md), Consequences),
  so E12's `e12-empty` `bko-nomap` contrast is expected `equivalent`. On holdout 0 the realized
  added rows' share of each mapped field sat within 0.004 of the label group's, sampling noise
  around that exact equality.
- The reading names the two models it reads: the regression and the boosting model.
- Unit [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md) reads this unit's E12
  runs beside its sweep as the fixed-fill point, with `compare --beside`.

## Acceptance

- `rows` reproduces each slice (`matches_folder=true`).
- The entry is `complete`.
- `make check` passes.
