---
id: privacy-value-microaggregation
title: Add the bko-micro arm, microaggregating numeric values over donors
status: ready
depends_on: --
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add the bko-micro arm, microaggregating numeric values over donors

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

With donor limits (arm `bko-capped`, ADR
[`research/sampler-donor-limits`](../docs/adr/research/sampler-donor-limits.md)), a numeric value
is still a verbatim copy of one record's. Averaging over several pattern-matched donors yields a
value no single record holds when the donors' values differ.

## Scope

**In.**

- `DrawSpec.microaggregate: int | None = None`, after `donors`, joining `_VARIANTS`; a value
  below 2 is refused.
- Arm `bko-micro`, `DrawSpec(min_donors=2, reuse_cap=1, microaggregate=3)`: 3 donors, on top of
  `bko-capped`'s limits.
- Every numeric field the draw copies is averaged except `age`: a unit holding a numeric field
  other than `age` is averaged, and every other unit is picked as `bko-capped` picks it. On E6's
  frame the averaged units are its eight count fields.
- An averaged unit's floor is `max(min_donors, k)`, its pool formed as `_pick_limited` forms it
  (`src/bko/generate/donors.py`). Its donors share their coded values, the truth of each
  `AtLeast` clause on its numeric fields and each numeric field's missingness, so a mean stays on
  its concept's side of the cutoff. An averaged pick with fewer than k free donors of a class
  counts that class as empty; with no class left it falls back to the label group, as an empty
  pool does. It logs through the existing `fallback` line (`src/bko/generate/direct.py`,
  `fallback holdout=%d fill=%s key=%s field=%s draws=%d%s presence=%s`), one line per averaged
  field, with no new format.
- Coded fields come from the primary donor. A numeric field is the k donors' mean, rounded
  half-even to a whole number for an integer field and to `frame.fraction_digits` otherwise.
- The weighted cap: counts are kept in k-ths, and a pick of weight `w` is admitted when
  `used + w <= reuse_cap * k`. The primary of a unit holding a coded field weighs k; a donor
  entering only a mean counts one k-th against the cap (k = 3 for `bko-micro`); in a unit of
  numeric fields only, each of its k donors weighs 1.
- `Draw.averaged`, default empty, naming every donor id of an averaged field: `trace` writes it
  only when non-empty, `privacy`'s donor reuse counts every id, and `attribution` attributes an
  averaged field `donor-mean`.
- One `micro-field holdout=%d fill=%s key=%s field=%s values=%d equal=%d primary=%d sse=%s sst=%s`
  line per cell and averaged field when the spec is set.
- ADR [`research/sampler-donor-limits`](../docs/adr/research/sampler-donor-limits.md) revised in
  place, stating that a mean of 3 leaves a count field about one third of its donors' variance,
  what the mean does not protect, and how the arm differs from SMOTE; ADR
  [`research/record-explainability`](../docs/adr/research/record-explainability.md) gains the
  `donor-mean` attribution.
- Runs `e20-thin` and `e20-empty`, each sliced in ten, `E9_THIN` and `E9_EMPTY` with arms
  `none`, `real`, `bko-direct`, `bko-capped` and `bko-micro`, candidate `bko-micro`, comparators
  `bko-direct`, `bko-capped` and `none`, `REFERENCE` `bko-direct`; the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry E20's Configuration with status `not yet run`,
  its primary contrast `bko-micro` minus `bko-capped` on `bko-capped`'s target measure.

**Out.**

- Perturbing coded fields.
- Averaging `age`, or only the numeric fields of mapped groups.
- A count of one per mean donor, two donors under a strict cap, and a k-anonymity claim.
- Any change to `bko-direct`'s or `bko-capped`'s draw: `DRAW_DIGEST`
  (`tests/test_generate_direct.py`) and `ADDED_DIGEST` (`tests/test_evaluate_direct.py`) stay
  byte-identical.

## Acceptance

- On a fixture, every microaggregated value is on its concept's side of the cutoff:
  `tests/test_generate_micro.py::test_a_microaggregated_value_stays_on_its_concepts_side_of_the_cutoff`.
- On a fixture whose donors' mean differs from each of their values, the microaggregated value
  equals none of them: `tests/test_generate_micro.py::test_a_mean_differs_from_each_donor_on_a_spread_fixture`.
- On a fixture whose pool repeats a record, the draw still averages distinct donor records:
  `tests/test_generate_micro.py::test_a_pool_that_repeats_a_record_still_averages_distinct_donors`.
- On a fixture, each micro donor's use counts against the reuse cap:
  `tests/test_generate_micro.py::test_a_mean_only_donor_counts_one_kth` and
  `test_a_donor_used_twice_in_means_is_refused_as_a_primary`.
- An own pool with fewer than k free donors falls back to the label group with no
  `SamplingError`: `tests/test_generate_micro.py::test_an_own_pool_with_fewer_than_k_free_donors_falls_back`
  and `test_a_two_record_cell_falls_back_at_k_3`.
- `age` is copied and an integer mean rounds half-even:
  `tests/test_generate_micro.py::test_age_is_copied_not_averaged` and
  `test_an_integer_mean_rounds_half_even`.
- The trace names every donor and donor reuse counts every averaged id:
  `tests/test_evaluate_micro.py::test_the_trace_names_every_donor`,
  `test_donor_reuse_counts_every_averaged_id` and `test_an_averaged_field_is_attributed_donor_mean`.
- `bko-capped`'s pinned draw is unchanged:
  `uv run --no-sync pytest tests/test_generate_capped_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null`
  passes.
- `RUNS` holds `e20-thin` and `e20-empty`: `tests/test_evaluate_micro_runs.py` and
  `tests/test_named_runs_slices.py` pass, its `KEYS` gaining the new names immediately before
  `"p1-id"` and its slice parametrize list gaining `e20`.
- One real slice fills for free, as evidence recorded in the PR body rather than a gate check:
  `python -m bko.evaluate score e20-thina` into `scratchpad/` exits 0 with `micro-field` lines
  for the eight count fields and no `refused` line.
- `make check` passes.
