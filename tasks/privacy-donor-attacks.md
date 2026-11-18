---
id: privacy-donor-attacks
title: Add a field-group match attack and a donor-reuse measure to the privacy reading
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: privacy-donor-limits, generate-coherent-free-fields
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add a field-group match attack and a donor-reuse measure to the privacy reading

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The distance attack barely sees a copied field group, and E8 reads `leaks_more=true` for
`bko-direct` on the thin cell
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e8-membership-inference-nndr-and-dcr-on-e6)). An attack
that reads verbatim field-group copies, and a count of how often each training record donates,
measure what the direct arms expose.

## Scope

**In.**

- A new module beside `bko.evaluate.membership` holding:
  - a field-group match attack for every arm, where a target's score is the most field groups any
    added row copies from it verbatim;
  - a donor-reuse measure, the maximum and 95th-percentile donations per training record per
    fill, read from `Draw.sources` (`bko.generate.direct`).
- Both join `privacy`'s output.
- ADR
  [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)
  revised in place.
- The reading over the E6 and E10 stores, the PRAM arms included, recorded as Runs rows of the
  [E8](../docs/EVIDENCE.md#e8-membership-inference-nndr-and-dcr-on-e6) and
  [E10](../docs/EVIDENCE.md#e10-identifier-perturbation) entries.

**Out.**

- Shadow-model and density-ratio attacks.
- The `REFERENCE` table's shape.

## Acceptance

- On fixtures, a planted verbatim copy scores AUC 1.0 and disjoint rows score 0.5.
- `membership.py`, where `privacy` assembles its output, stays under its `make sizes` cap.
- `make check` passes.
