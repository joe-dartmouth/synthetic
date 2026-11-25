---
id: evaluate-variant-dev-results
title: Run the direct variants on the development frames and apply the inclusion rule
status: blocked
depends_on: generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-value-microaggregation, generate-coherent-free-fields
blocks: evaluate-composite-arm
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

- The development runs registered by units
  [`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md),
  [`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md),
  [`privacy-value-microaggregation`](privacy-value-microaggregation.md) and
  [`generate-coherent-free-fields`](generate-coherent-free-fields.md), and `bko-capped`'s runs
  `e14-thin` and `e14-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e14-donor-floor-and-reuse-cap-on-the-diabetes-130-removal)).
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

## Acceptance

- Each entry is `complete` and states pass or fail under the rule.
- `make check` passes.
