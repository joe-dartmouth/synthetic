---
id: privacy-value-microaggregation
title: Add the bko-micro arm, microaggregating numeric values over donors
status: blocked
depends_on: privacy-donor-limits
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add the bko-micro arm, microaggregating numeric values over donors

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

With donor limits (unit [`privacy-donor-limits`](privacy-donor-limits.md)), a numeric value is
still a verbatim copy of one record's. Averaging over several pattern-matched donors yields a
value no single record holds when the donors' values differ.

## Scope

**In.**

- `DrawSpec.microaggregate: int | None`.
- Arm `bko-micro`, at 3 donors, on top of `bko-capped`'s limits.
- A numeric field of a group is the mean of that many distinct pattern-matched donor records'
  values, which stays on the concept's side of its cutoff.
- Each micro donor counts against `bko-capped`'s reuse cap.
- Coded fields are unchanged.
- The trace names every donor.
- The ADR of unit [`privacy-donor-limits`](privacy-donor-limits.md) revised in place.
- Development `RUNS` entries and the [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's
  Configuration with status `not yet run`, with the same target measure as `bko-capped`.

**Out.**

- Perturbing coded fields.

## Acceptance

- On a fixture, every microaggregated value is on its concept's side of the cutoff.
- On a fixture, a microaggregated value equals no single donor's value when the donors differ.
- On a fixture whose pool repeats a record, the draw still averages distinct donor records.
- On a fixture, each micro donor's use counts against the reuse cap.
- `make check` passes.
