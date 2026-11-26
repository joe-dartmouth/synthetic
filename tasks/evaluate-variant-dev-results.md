---
id: evaluate-variant-dev-results
title: Run the direct variants on the development frames and apply the inclusion rule
status: blocked
depends_on: privacy-value-microaggregation
blocks: evaluate-composite-arm, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the direct variants on the development frames and apply the inclusion rule

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Each variant claims one improvement over `bko-direct`. The runs say which hold without costing
removed-cell recall, and the variant inclusion rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md)
says which enter `bko-direct-2`.

## Scope

**In.**

- The development runs registered by unit
  [`privacy-value-microaggregation`](privacy-value-microaggregation.md), `bko-capped`'s runs
  `e14-thin` and `e14-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e14-donor-floor-and-reuse-cap-on-the-diabetes-130-removal)),
  `bko-tree`'s runs `e15-thin` and `e15-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e15-ancestor-shrunk-shares-on-the-diabetes-codes-frame)),
  the `bko-hier` arms' runs `e16-thin` and `e16-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e16-conditional-is_a-draws-on-the-diabetes-130-removal)),
  and `bko-coherent`'s runs `e17-thin` and `e17-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e17-coherent-free-fields-on-the-diabetes-130-removal)).
- For each arm, `contradictions --rows` (ADR
  [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md),
  Decision 2), `attribution` (ADR
  [`research/record-explainability`](../docs/adr/research/record-explainability.md), Decision 7),
  and `privacy`'s match attack and
  donor reuse (ADR [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entries' Runs, Result and Read against.
- For each variant, whether it meets the variant inclusion rule.

**Out.**

- Choosing parameters not registered.

**Findings from planning.**

- E20's primary contrast is `bko-micro` minus `bko-capped`.
- `bko-micro`'s count fields carry about one third of their donors' variance, as ADR
  [`research/sampler-donor-limits`](../docs/adr/research/sampler-donor-limits.md), once unit
  `privacy-value-microaggregation` revises it, states; its fidelity reading is read with that.

## Acceptance

- Each entry is `complete` and states pass or fail under the rule.
- `make check` passes.
