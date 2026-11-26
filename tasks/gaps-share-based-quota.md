---
id: gaps-share-based-quota
title: Recommend a cell's row count from coverage and expected size
status: ready
depends_on: --
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

- A new `bko.gaps.quota`, importing nothing from `bko.generate` or `bko.evaluate`, holding:
  - `coverage(share)`: the smallest n at which every concept whose fused share is at least 0.02
    (`SHARE_FLOOR`) appears with probability at least 0.95, by a union bound (the sum over those
    concepts of (1-p)^n is at most `MISS_CHANCE` 0.05); 0 when no concept qualifies;
  - `expected_size(cell, counts)`: for cell (a, b, y), n(a, b, not y) times (N1 + 1/2) / (N0 + 1/2),
    N1 and N0 the counts of (a, b', y) and (a, b', not y) over every other value b' of the
    second cell axis, read by position; when both are 0, the split's counts of y and not y. An
    exact `Fraction`, rounded half up;
  - `recommended(count, covered, expected)`: `max(0, max(covered, expected) - count)`;
  - `QuotaRule`, the seven rules: `fill-to` (the current rule, `fill_to - count`), and six counts,
    `coverage`, `coverage-x2`, `coverage-x4`, `coverage-x8` (two, four and eight times
    coverage), `expected-size` and `recommended`. Every rule but `recommended` asks
    `max(0, value - count)`; a cell asked 0 leaves the request, and a request left empty raises
    the existing "nothing to fill" `EvaluationError`.
- `Experiment.quota_rule: QuotaRule = QuotaRule.FILL_TO`, after `fill_to`, refused beside
  `flagged_cell` or `synthetic_share`, emitted in the `experiment` line as `quota_rule=<value>`
  only when it is not the default, with `ASSUMED["quota_rule"] = "fill-to"`
  (`bko.evaluate.run_folder`).
- `gap_request` takes the data's field rows and dataset name; a non-default rule with no rows is
  refused. Its five callers pass both: `run`, `membership`, `shared`, `trace` and
  `added_contradictions` under `src/bko/evaluate/`.
- A `quota-rule holdout=%d fill=%s key=%s rule=%s count=%d coverage=%d expected=%d asked=%d`
  line per cell under a non-default rule, `fill` and `key` as the direct arms' lines name them.
- The representativeness reading,
  `python -m bko.evaluate representativeness <run> --rows <store>...`, its subcommand in
  `src/bko/evaluate/reader_commands.py` and `src/bko/evaluate/__main__.py`, writing
  `representativeness.csv` and `representativeness.log`: per holdout, arm and flagged cell, the
  largest absolute gap between the added rows' concept shares and the fused shares, and the
  cell's share of the arm's training set beside `real`'s. Removed-cell recall is reported beside
  it. Its doc homes: an `AGENTS.md` Commands entry, the `docs/DEVELOPMENT.md`
  Runs table rows and Evaluation paragraph, its entries in `ROLES` (`src/bko/app/runs.py`) and
  `tests/test_run_folders.py`.
- A new ADR under `docs/adr/research/` that records the rule, notes that `bko.evaluate.quotas` is
  a different thing, and states that on E6's cell `expected-size` and `recommended` ask within 30
  rows of the fixed fill; one sentence in ADR
  [`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md).
- E21's sweep: E12's `e12-thin` and `e12-empty` with `quota_rule` set to each of the six
  non-default rules, sliced in ten, `REFERENCE` `real`. Every E12 arm is free, so the sweep makes
  no paid call; E6's own `RUNS` arms include paid Messages arms and are not swept. The fixed
  fill is not registered again: it is read from E12's runs with `compare --beside`.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry E21's Configuration with status
  `not yet run`, the representativeness measure fixed as its reading.

**Out.**

- A stopping rule inside the sampler.
- An external reference population.
- Changing which cells are flagged.
- An independence product or a per-concept bound in place of the union bound.
- Registering the fixed fill again.
- Running E21: unit [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md).

## Acceptance

- Each function gives a hand-computed count on a fixture: `tests/test_gaps_quota.py`, one test
  per hand fixture (coverage {0.5, 0.1, 0.015} -> 29, {0.02} -> 149, {} -> 0; expected size
  10, 6, 12 -> 5; recommended at count 2, 29, 5 -> 27) and
  `test_the_multiples_are_coverage_times_2_4_and_8`.
- A cell of no record gets a finite count from each function:
  `tests/test_gaps_quota.py::test_a_cell_of_no_record_gets_a_finite_count_from_each_function`.
- `rows` on one existing run folder on E6's frame still reproduces it: `rows e10-pilot-thin`
  over `runs/e10-3` (E10's pilot) writes a `rows.jsonl` that `cmp` finds equal to
  `runs/e10-3-rows/rows.jsonl`, with `matches_folder=true` in its log, at the unit's base and at
  its tip.
- The rule runs and every reader of a run folder still reads it:
  `tests/test_evaluate_quota_rule.py` passes, with
  `test_an_older_experiment_line_reads_at_the_assumed_quota_rule`,
  `test_a_coverage_rule_run_scores_end_to_end`, `test_shared_rescores_a_run_under_a_share_rule`,
  `test_trace_reads_a_run_under_a_share_rule` and
  `test_contradictions_reads_a_run_under_a_share_rule`.
- The reading measures a fill against the fused shares:
  `tests/test_evaluate_representativeness.py::test_the_gap_is_the_largest_share_difference`.
- `RUNS` holds E21: `tests/test_evaluate_quota_runs.py` passes with
  `test_each_sweep_run_differs_from_e12_only_in_its_rule` and `test_no_sweep_arm_is_paid`.
- `make check` passes.
