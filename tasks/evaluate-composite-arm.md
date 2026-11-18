---
id: evaluate-composite-arm
title: Add the bko-direct-2 arm, composing the variants that passed and the supported quota rule
status: blocked
depends_on: evaluate-variant-dev-results, evaluate-quota-sweep-results
blocks: evaluate-composite-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add the bko-direct-2 arm, composing the variants that passed and the supported quota rule

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The variants were each read alone. The method is their composition.

## Scope

**In.**

- Arm `bko-direct-2`. It takes the `DrawSpec` fields and the `Change`
  (`bko.evaluate.perturb.Change`) of every variant unit
  [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md) records as passing.
- The quota rule that unit
  [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md)'s reading supports: the
  smallest registered count within the measure's tolerance.
- Development `RUNS` entries on E6's and E3's frames, against `bko-direct`, and the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`.
- The ADR unit [`plan-direct-method-research`](plan-direct-method-research.md) adds, revised in
  place with the composition.

**Out.**

- A variant or value unit
  [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md) did not record as passing.

## Acceptance

- A fixture draw of `bko-direct-2` satisfies each included variant's own fixture property.
- `bko-direct`'s digests are unchanged.
- `make check` passes.
