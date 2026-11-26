---
id: evaluate-p1-partial-replicas
title: Score a partial replica as a partial gap fill is scored, under new P1 run names
status: ready
depends_on: --
blocks: evaluate-tableshift-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Score a partial replica as a partial gap fill is scored, under new P1 run names

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The [P1 entry](../docs/EVIDENCE.md#p1-tableshift-readmission-split) records runs `p1-1`
(`p1-id`) and `p1-2` (`p1-ood`): each exits 2 on holdout 0, when the `tvae-replica` fill is
refused after 5 attempts. TVAE's rejection sampler draws 1,000,000 rows and still lacks rows of
TableShift's rare cells, among them the `<1` age band, `1-1/male` and `>=8/male`. The `tvae` gap
fill on the same holdout fills partially on its last attempt, with its `shortfall` lines, as ADR
[`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md) gives it; a
replica fill refuses instead, because its scores read it as a copy of every training cell. The
seeds and the registration are fixed, so a rerun of `p1-id` or `p1-ood` stops at the same fill.

The decision taken on 2026-11-26: a P1 replica that a generator cannot complete is scored as the
rows it kept, with its shortfall logged, as a partial gap fill is, rather than stopping the run.
An existing `RUNS` entry never changes (`direct-index.md`, What every unit relies on), so the
rule runs under new run names.

## Scope

**In.**

- `Experiment.partial_replica: bool = False`, after `replica_rows`, refused when `True` with no
  `replica_arms`, emitted as `partial_replica=true`, with `ASSUMED["partial_replica"] = "false"`:
  the field that lets a replica fill on its last attempt fill partially through
  `bko.evaluate.shortfall.partial_fill`, as a gap fill does: the kept rows checked against the
  replica request less the shortfall, a `shortfall` line per short cell, and the `arm` line's
  params naming the shortfall. Attempts before the last still retry on a `ShortfallError`, and a
  replica takes no early partial fill on structural strata, as now. Its default keeps the
  current refusal; the `experiment` line emits the field only when it is not the default, with
  its default in `ASSUMED` (`bko.evaluate.run_folder`), so every existing run folder still
  passes its identity checks. `fill_arm` (`src/bko/evaluate/filling.py`) takes
  `partial: bool = False`, its last-attempt partial reading `kind == "gap" or partial`, its
  structural early partial staying gap-only, and the replica call in `src/bko/evaluate/run.py`
  passes `partial=experiment.partial_replica`.
- A direct arm's short replica still refuses: `DirectArm` raises `BaselineError` when a draw
  leaves cells short (`src/bko/evaluate/direct.py:240-242`), whatever the field says, so the rule
  reaches baseline replicas such as `tvae-replica` only.
- A partial replica is marked per holdout where a reader of the scores sees it: `scores.csv`
  gains a `replica-short` measure row (`Measure.REPLICA_SHORT`, source `replica`), scope `all`,
  its value the rows missing, for each replica whose `Baseline.params` holds `short`, and
  `compare` logs `replica-short holdout=%d arm=%s rows=%d` for each such row. A run whose
  replicas all complete writes no such row. `merge` drops `replica-short` keys before it compares
  holdouts' key sets, so slices where only some hold the row still merge.
- The re-draw `rows` makes of a partial replica goes through `fill_arm`, the helper the draw
  uses, so its sha256 matches the `arm` line's and it logs the same `shortfall` lines the draw
  logged. `trace` re-draws no baseline (`src/bko/evaluate/trace.py:20`): for a partial baseline
  replica it writes the method, the seed and the `arm` line's `short` params, without a re-draw,
  and logs an INFO line `trace replica-short holdout=%d arm=%s short=%s`.
  `panel`, `privacy` and `syntheval` read the fill from the `rows` store and read the shortfall
  from the `arm` line's params, logging no `shortfall` line, as `panel` reads a partial gap fill
  now. `shared` refuses a folder whose replica `arm` line holds `short`, as it refuses a partial
  gap fill.
- `bko.evaluate.runs_published` gains `P1_PARTIAL`, `P1` with the field set, and
  `bko.evaluate.named_runs.RUNS` gains `p1-id-partial` (dataset `tableshift-id`) and
  `p1-ood-partial` (dataset `tableshift-ood`), in that order, after `p1-ood` and before `p2`.
  `p1-id` and `p1-ood` stay as they are.
- The tests that pin `RUNS`'s order gain the two names: `KEYS` in
  `tests/test_named_runs_slices.py` between `"p1-ood"` and `"p2"`, and the P1 slice assertion in
  `tests/test_evaluate_fixed_partition.py`, which reads `list(RUNS)[-3:-1]` and becomes the four
  P1 names at `[-5:-1]`.
- ADR `research/e1-evaluation-protocol`: the partial replica rule for an experiment that sets the
  field, and what a replica reading means on fewer rows where it now reads a replica as a copy
  of every training cell, with the `replica-short` row as the mark; ADR
  `research/tableshift-comparison`: the new run names. Each with `updated` bumped and a dated
  Ruled Out row for the reading it replaces.
- The [P1 entry](../docs/EVIDENCE.md#p1-tableshift-readmission-split)'s Configuration names
  `p1-id-partial` and `p1-ood-partial` as the runs to run, with status `not yet run`, and
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation)'s paragraph on what `RUNS` holds,
  which names P1's runs, names them too.

**Out.**

- Running the runs: unit [`evaluate-tableshift-results`](evaluate-tableshift-results.md) runs
  them.
- Any change to P1's split, arms, seeds, replica row count or row limit, or to the gap-fill rule.
- A partial replica in any run other than the two new ones.
- A partial fill for a direct arm's replica, and a `trace` re-draw of a baseline.

## Acceptance

- With a fake replica runner that raises a `ShortfallError` on every attempt, an experiment that
  sets the field retries each attempt before the last, logging `retry ... reason=`, then scores
  the kept rows on the last, logging a `shortfall` line per short cell whose `arm=` is
  `<arm>-replica`, and its `arm` line's params name the shortfall. The default experiment still
  logs `refused holdout=%d arm=%s attempts=%d` and raises:
  `tests/test_evaluate_partial_replica.py::test_a_partial_replica_retries_each_attempt_before_the_last_then_fills_partially`
  and `test_the_default_still_refuses`, reusing `ONLY_SMOTE` and `shorting` of
  `tests/test_evaluate_run.py`.
- A replica takes no early partial fill on structural strata:
  `tests/test_evaluate_partial_replica.py::test_a_replica_takes_no_early_partial_on_structural_strata`.
- A replica that completes scores identically with the field set and with its default:
  `tests/test_evaluate_partial_replica.py::test_a_replica_that_completes_scores_identically_with_the_field_set`.
- A partial replica writes its `replica-short` row in `scores.csv`, and `compare` prints its
  holdout and arm; a run whose replicas complete writes none:
  `tests/test_evaluate_partial_replica.py::test_a_partial_replica_writes_a_replica_short_row` and
  `test_a_run_whose_replicas_complete_writes_no_such_row`, and
  `tests/test_evaluate_partial_replica_readers.py::test_compare_prints_each_partial_holdout_and_arm`.
- `merge` accepts slices where only some hold a `replica-short` row:
  `tests/test_evaluate_partial_replica_readers.py::test_merge_accepts_slices_where_only_some_hold_replica_short`.
- `rows`, re-drawing a partial replica, reproduces the `arm` line's sha256 and logs the draw's
  `shortfall` lines; `panel` over the `rows` store logs the replica in its `partial=` count and no
  `shortfall` line; `trace` of a partial baseline replica writes its method, seed and the `arm`
  line's shortfall without a re-draw and logs its `trace replica-short` line; `shared` refuses a
  partial replica:
  `test_rows_reproduces_a_partial_replicas_sha256_and_shortfall_lines`,
  `test_panel_counts_the_replica_in_partial`,
  `test_trace_writes_a_partial_baseline_replicas_shortfall` and
  `test_shared_refuses_a_partial_replica` in `tests/test_evaluate_partial_replica_readers.py`.
- `score --replay` and `score --rehearse` of a partial replica log its `shortfall` lines as the
  run did:
  `tests/test_evaluate_partial_replica_readers.py::test_replay_and_rehearse_log_the_shortfall_lines`.
- Every test above that reads a `shortfall` line asserts the `partial_fill` key set (`holdout`,
  `arm`, `attempt`, `cell`, `short`, `quota`), so a `give_up` line cannot pass
  (`cleanup:evaluate/shortfall-line-two-formats` in [`CLEANUP.md`](CLEANUP.md)).
- `P1_PARTIAL`'s `experiment` line emits the field, and an older line reads at the assumed
  default: `tests/test_evaluate_partial_replica.py::test_the_experiment_line_names_partial_replica_only_when_set`
  and `test_an_older_experiment_line_reads_at_the_assumed_partial_replica`; the field without
  `replica_arms` is refused: `test_a_partial_replica_without_replica_arms_is_refused`.
- The default experiment's `experiment` line is byte-identical to the line at this unit's base:
  `tests/test_evaluate_partial_replica.py::test_the_default_experiment_line_is_unchanged`, a
  literal captured at the base and committed green before the field is added.
- `P1_PARTIAL` with the field set back to its default equals `P1`, and `RUNS["p1-id"]` and
  `RUNS["p1-ood"]` are unchanged: `tests/test_evaluate_fixed_partition.py::test_p1_partial_with_the_field_at_its_default_equals_p1`
  and `test_the_p1_runs_are_unchanged`, with `list(RUNS)[-5:-1]` the four P1 names.
- `RUNS["p1-id-partial"]` and `RUNS["p1-ood-partial"]` read their datasets, and `KEYS` holds them
  between `"p1-ood"` and `"p2"`: `tests/test_named_runs_slices.py` passes.
- `make check` passes.
