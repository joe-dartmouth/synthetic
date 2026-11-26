---
id: evaluate-p1-partial-replicas
title: Score a partial replica as a partial gap fill is scored, under new P1 run names
status: ready
depends_on: --
blocks: evaluate-tableshift-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/e1-evaluation-protocol, research/tableshift-comparison
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

- An `Experiment` field that lets a replica fill on its last attempt fill partially through
  `bko.evaluate.shortfall.partial_fill`, as a gap fill does: the kept rows checked against the
  replica request less the shortfall, a `shortfall` line per short cell, and the `arm` line's
  params naming the shortfall. Attempts before the last still retry on a `ShortfallError`, and a
  replica takes no early partial fill on structural strata, as now. Its default keeps the
  current refusal; the `experiment` line emits the field only when it is not the default, with
  its default in `ASSUMED` (`bko.evaluate.run_folder`), so every existing run folder still
  passes its identity checks.
- A partial replica is marked per holdout where a reader of the scores sees it: `scores.csv`
  gains a `replica-short` measure row, scope `all`, its value the rows missing, for each replica
  that filled partially, and `compare` prints each such holdout and arm. A run whose replicas
  all complete writes no such row.
- The re-draw `rows` and `trace` make of a partial replica goes through `fill_arm`, the helper
  the draw uses, so its sha256 matches the `arm` line's and it logs the same `shortfall` lines
  the draw logged. `panel`, `privacy` and `syntheval` read the fill from the `rows` store and
  read the shortfall from the `arm` line's params, logging no `shortfall` line, as `panel` reads
  a partial gap fill now. `shared` refuses a partial replica as it refuses a partial gap fill.
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

## Acceptance

- With a fake replica runner that raises a `ShortfallError` on every attempt, an experiment that
  sets the field retries each attempt before the last, logging `retry ... reason=`, then scores
  the kept rows on the last, logging a `shortfall` line per short cell whose `arm=` is
  `<arm>-replica`, and its `arm` line's params name the shortfall. The default experiment still
  logs `refused holdout=%d arm=%s attempts=%d` and raises.
- A replica that completes scores identically with the field set and with its default.
- A partial replica writes its `replica-short` row in `scores.csv`, and `compare` prints its
  holdout and arm; a run whose replicas complete writes none.
- `rows` and `trace`, re-drawing a partial replica, reproduce the `arm` line's sha256 and log the
  draw's `shortfall` lines; `panel` over the `rows` store logs the replica in its `partial=`
  count and no `shortfall` line.
- `score --replay` and `score --rehearse` of a partial replica log its `shortfall` lines as the
  run did.
- `P1_PARTIAL`'s `experiment` line emits the field, and a run folder holding that line passes
  `run_folder`'s identity checks under `rows`, `panel` and `trace`.
- The default experiment's `experiment` line is byte-identical to the line at this unit's base.
- `P1_PARTIAL` with the field set back to its default equals `P1`.
- `RUNS["p1-id-partial"]` and `RUNS["p1-ood-partial"]` read their datasets, and `RUNS["p1-id"]`
  and `RUNS["p1-ood"]` equal their entries at this unit's base.
- `make check` passes.
