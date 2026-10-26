---
id: evaluate-holdout-count
title: E1 on more holdouts
status: blocked
depends_on: evaluate-e1-second-configuration
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# E1 on more holdouts

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

On 100 holdouts every primary contrast's corrected interval spans about plus or minus 0.03 of
recall, and the flagged subgroups hold about five positive test records per holdout, so E1 reads
inconclusive. More holdouts narrow the intervals for free on the baselines and through new calls on
the BKO arms, up to the floor the corrected variance sets as splits overlap.

## Scope

**In.**

- `E1.holdout_seeds` extended to 300, with the first 100 unchanged so `e1-3`'s log still replays
  them.
- First, free: the baselines and controls on 300 holdouts, and how each contrast's corrected
  interval width moves from 100 to 300, read against the width the Nadeau-Bengio correction allows
  at these split sizes; if the widths do not narrow enough to change a verdict, the record says so
  and the paid part does not run.
- Then, paid: the BKO arms on the 200 added holdouts, the first 100 answered from the judged run's
  log through `--resume`, at an estimate of twice the judged run's live spend, stated before the
  run; detached, kept awake, on mains power and watched as `e1-3` was.
- A new E1 Runs row, Result and Read against in `docs/EVIDENCE.md` as the judged configuration's
  reading on 300 holdouts.

**Out.** Changing the split rule or the test share.

## Acceptance

- The free reading regenerates from a committed command; the paid run is recorded with its spend,
  or the entry states the width measure that stopped it.
- `make check`.
