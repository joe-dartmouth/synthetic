---
id: evaluate-dose-response
title: Fill above the threshold
status: ready
depends_on: --
blocks: evaluate-diabetes-subgroup-removal
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# Fill above the threshold

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E1 fills each underrepresented cell up to the threshold of 20, so the `bko` arm adds about 20 rows
to 238, and a 20-row dose bounds any effect it can show. Filling to 30, 40 and 80 rows per cell, every
row-adding arm at the same quota, shows whether more rows move recall and AUC-PR, and in which
direction.

## Scope

**In.**

- `Experiment.fill_to` split in two: the threshold that flags a cell, which stays the one of
  [`docs/RESEARCH.md`](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting) and decides
  the flagged subgroups, and a fill target each flagged cell is filled up to, the threshold by
  default; a test pins that `E1`'s flagged set and requests are unchanged.
- Three named runs (`bko.evaluate.named_runs.RUNS`) on the judged configuration, targets 30, 40 and 80, each with
  the baselines and the `bko` arm, its gap and replica fills live, since replay pairs calls by
  order and a changed gap request differs at the first call. Spend estimate per run: one arm's
  share of the judged run's spend, its gap-fill part scaled by the rows the target asks over the
  rows the threshold asks, stated in the Runs row before the run, under the run's ceiling and the
  paid-run protocol (`docs/DEVELOPMENT.md`, Evaluation).
- Per target: each arm's measures, each contrast's corrected interval, validity, drops, privacy and
  test copies, in a new `docs/EVIDENCE.md` entry that reads the judged run as the target-20 point.

**Out.** Changing the threshold; a verdict (the criteria read the judged configuration at its
threshold).

## Acceptance

- Each target's measures regenerate from `score <run>` and `compare <run>`, and its run is
  recorded with its spend and replays byte for byte.
- `make check`.
