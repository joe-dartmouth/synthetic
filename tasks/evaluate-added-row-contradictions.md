---
id: evaluate-added-row-contradictions
title: Count contradictions and coherence breaks in each arm's added rows
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: generate-is-a-conditional-draws, generate-coherent-free-fields
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Count contradictions and coherence breaks in each arm's added rows

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The `contradictions` command reads training splits only (ADR
[`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md),
Decision 2). No reading shows whether an arm's added rows contradict the ontology or break the
Diabetes coherence rules that `bko.evaluate.perturb.incoherent` checks. Units
[`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md) and
[`generate-coherent-free-fields`](generate-coherent-free-fields.md) take this count as their target
measure.

## Scope

**In.**

- A function over a `rows.jsonl` store that counts, per arm and fill, the five kinds of ADR
  [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md)
  and the three rules of `bko.evaluate.perturb.incoherent`.
- A `contradictions --rows` reading that runs it.
- The ADR's Decision 2 revised in place to name the added rows as a second basis.
- The reading over the E6 stores, recorded as a Runs row of
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal).

**Out.**

- Correcting any row.

## Acceptance

- Fixture rows with one planted violation of each kind are counted exactly.
- `make check` passes.
