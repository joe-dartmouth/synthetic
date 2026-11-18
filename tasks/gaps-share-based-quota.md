---
id: gaps-share-based-quota
title: Recommend a cell's row count from coverage and expected size
status: blocked
depends_on: refactor-evaluate-size-splits, evaluate-control-runs-registered
blocks: evaluate-quota-sweep-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/share-based-quota
---

# Recommend a cell's row count from coverage and expected size

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The harness asks each flagged cell for `fill_to - count` rows
(`bko.evaluate.experiment.gap_request`), a constant unrelated to the model. A count derived from
the fused shares and the neighbouring cells says how many rows a cell needs to be representative.

## Scope

**In.**

- A new `bko.gaps.quota` holding:
  - `coverage(cell)`: the rows at which every concept whose fused share is at least 0.02 appears
    with probability 0.95;
  - `expected_size(cell)`: the cell's count predicted from its neighbours, the count of its age
    band and sex under the other label times the label odds of the same band under the other sex;
  - `recommended(cell)`: the larger of the two, less the cell's count.
- `Experiment.quota_rule`, defaulting to the current rule, emitted in the `experiment` line only
  when it is not the default, with its default in `ASSUMED` (`bko.evaluate.run_folder`).
- `gap_request` takes the field rows.
- A new ADR under `docs/adr/research/` that records the rule.
- A sweep family on E6's frame: each arm of the E6 family that unit
  [`evaluate-control-runs-registered`](evaluate-control-runs-registered.md) registers, at the fixed
  fill, the coverage count, the expected-size count and the recommended count. Those arms are all
  free, so the sweep makes no paid call; E6's own `RUNS` arms include paid Messages arms and are
  not swept. Three of those arms land in that unit's dependencies, so this unit depends on it.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`.
  It fixes the representativeness measure as the reading: the largest gap between a cell's
  realized and fused concept shares, and the cell's share of the training set against `real`'s.
  Removed-cell recall is reported beside it.

**Out.**

- A stopping rule inside the sampler.
- An external reference population.
- Changing which cells are flagged.

## Acceptance

- Each function gives a hand-computed count on a fixture.
- A cell of no record gets a finite count from each function.
- `rows` on one existing E6 slice still reproduces it.
- `make check` passes.
