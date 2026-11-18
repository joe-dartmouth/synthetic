---
id: evaluate-trace-attribution
title: Read what each arm kind's trace can attribute a record's fields to
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Read what each arm kind's trace can attribute a record's fields to

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`trace` writes the facts behind each record of one fill (ADR
[`research/record-explainability`](../docs/adr/research/record-explainability.md)). No reading
contrasts what each arm kind can attribute, so the explainability claim has no measure.

## Scope

**In.**

- A reading over `trace.jsonl` files that counts, per arm, each record field's attribution class:
  training donor id, knowledge fact, prompt section only, or method and seed only. It writes a CSV.
- ADR [`research/record-explainability`](../docs/adr/research/record-explainability.md) revised in
  place: the contrast claimed is attribution to knowledge facts, and donor attribution is not
  unique to the direct arms.
- The reading over one E6 holdout, recorded as a Runs row of
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal).

**Out.**

- A new trace format.
- Naming donors for the baselines.

## Acceptance

- Fixture traces of one arm of each kind give the expected classes.
- `make check` passes.
