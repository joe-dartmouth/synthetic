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
  params naming the shortfall. Its default keeps the current refusal; the `experiment` line
  emits the field only when it is not the default, with its default in `ASSUMED`
  (`bko.evaluate.run_folder`), so every existing run folder still passes its identity checks.
- The re-draw that `rows`, `panel`, `trace` and `privacy` read reproduces a partial replica
  through the same helper the draw uses, so its sha256 matches the `arm` line's.
- `bko.evaluate.runs_published` gains `P1_PARTIAL`, `P1` with the field set, and
  `bko.evaluate.named_runs.RUNS` gains `p1-id-partial` (dataset `tableshift-id`) and
  `p1-ood-partial` (dataset `tableshift-ood`). `p1-id` and `p1-ood` stay as they are.
- Tests that pin `RUNS`'s order or members gain the two names.
- ADR `research/e1-evaluation-protocol`: the partial replica rule for an experiment that sets the
  field, and what a replica reading means on fewer rows; ADR `research/tableshift-comparison`:
  the new run names. Each with `updated` bumped and a dated Ruled Out row for the reading it
  replaces.
- The [P1 entry](../docs/EVIDENCE.md#p1-tableshift-readmission-split)'s Configuration names
  `p1-id-partial` and `p1-ood-partial` as the runs to run, with status `not yet run`.

**Out.**

- Running the runs: unit [`evaluate-tableshift-results`](evaluate-tableshift-results.md) runs
  them.
- Any change to P1's split, arms, seeds, replica row count or row limit, or to the gap-fill rule.
- A partial replica in any run other than the two new ones.

## Acceptance

- With a fake replica runner that raises a `ShortfallError` on every attempt, an experiment that
  sets the field scores the kept rows and logs a `shortfall` line per short cell, and the default
  experiment still logs `refused` and raises.
- A partial replica's re-draw reproduces the `arm` line's sha256.
- The default experiment's `experiment` line is byte-identical to the line at this unit's base.
- `RUNS["p1-id-partial"]` and `RUNS["p1-ood-partial"]` read their datasets, and `RUNS["p1-id"]`
  and `RUNS["p1-ood"]` equal their entries at this unit's base.
- `make check` passes.
