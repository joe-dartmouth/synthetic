---
id: privacy-donor-limits
title: Add the bko-capped arm, with a donor floor and a reuse cap
status: ready
depends_on: --
blocks: privacy-value-microaggregation, evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/sampler-donor-limits
---

# Add the bko-capped arm, with a donor floor and a reuse cap

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A thin cell's few members donate many times each, so `bko-direct`'s rows there repeat the same
records' values. A floor on donors and a cap on reuse bound that exposure.

## Scope

**In.**

- `DrawSpec.min_donors` and `DrawSpec.reuse_cap`.
- Arm `bko-capped`, at `min_donors=2` and `reuse_cap=1`.
- A pattern with fewer cell donors than the floor reads the label group.
- A donor at the cap in a fill is skipped.
- A new ADR under `docs/adr/research/` that records that noised shares give no guarantee while
  values are copied, and how E10's PRAM arms
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e10-identifier-perturbation)) relate.
- Development `RUNS` entries against `bko-direct`, and the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`. Its
  target measure is `privacy`'s match-attack AUC (ADR
  [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)).

**Out.**

- Differential-privacy claims.
- Distance-based rejection.

## Acceptance

- Fixture draws never exceed the cap.
- Fixture draws never copy a singleton pattern from the cell.
- `make check` passes.
