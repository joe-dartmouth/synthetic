---
id: generate-coherent-free-fields
title: Add the bko-coherent arm, drawing free fields by sequential CART
status: ready
depends_on: --
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/free-field-synthesis
---

# Add the bko-coherent arm, drawing free fields by sequential CART

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`bko-direct` takes each free field from a different record, which breaks the Diabetes drug rules
in a share of its rows that the
[`docs/IDEAS.md`](../docs/IDEAS.md#coherent-free-field-draws-for-bko-direct) entry "Coherent
free-field draws for `bko-direct`" measures. Copying the free
fields from one record instead would copy every unmapped field of one patient, which
`privacy`'s match attack (ADR [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)) reads. A conditional model
keeps the fields coherent without a joint copy.

## Scope

**In.**

- Arm `bko-coherent`: free fields drawn by the in-house sequential CART
  (`bko.baselines.classical`), conditioned on the cell and the drawn mapped fields.
- The CART draw is plugged in as a `Change` (`bko.evaluate.perturb`) through
  `DirectArm.fill(change=...)` (`bko.evaluate.direct`), since `bko.generate` imports nothing from
  `bko.baselines`.
- The trace names the leaf donor per field.
- A new ADR under `docs/adr/research/` that records the draw.
- Development `RUNS` entries and the [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's
  Configuration with status `not yet run`. Its target measure is the coherence counts of
  `contradictions --rows` (ADR [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md), Decision 2).
- The IDEAS entry "Coherent free-field draws for `bko-direct`" deleted.

**Out.**

- A joint copy from one record.
- Rule-based rejection.

## Acceptance

- In a fixture fill, each free field names a leaf donor in the trace.
- `bko-direct`'s digests are unchanged.
- `make check` passes.
