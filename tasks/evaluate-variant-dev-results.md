---
id: evaluate-variant-dev-results
title: Run the direct variants on the development frames and apply the inclusion rule
status: blocked
depends_on: generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-donor-limits, privacy-value-microaggregation, generate-coherent-free-fields, evaluate-trace-attribution
blocks: --
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
  [`privacy-donor-limits`](privacy-donor-limits.md),
  [`privacy-value-microaggregation`](privacy-value-microaggregation.md) and
  [`generate-coherent-free-fields`](generate-coherent-free-fields.md).
- For each arm, the readings of units
  [`evaluate-added-row-contradictions`](evaluate-added-row-contradictions.md),
  [`privacy-donor-attacks`](privacy-donor-attacks.md) and
  [`evaluate-trace-attribution`](evaluate-trace-attribution.md).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entries' Runs, Result and Read against.
- For each variant, whether it meets the variant inclusion rule.

**Out.**

- Choosing parameters not registered.

## Acceptance

- Each entry is `complete` and states pass or fail under the rule.
- `make check` passes.
