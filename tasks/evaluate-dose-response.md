---
id: evaluate-dose-response
title: Fill above the threshold
status: blocked
depends_on: evaluate-e1-second-configuration
blocks: --
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
to 238, and a 20-row dose bounds any effect it can show. Filling to 30 and 40 rows per cell, every
row-adding arm at the same quota, shows whether more rows move recall and AUC-PR, and in which
direction.

## Scope

**In.**

- A fill target in `E1`, the threshold by default: every row-adding arm fills each underrepresented
  cell up to it, so the comparison stays at equal row counts, and the threshold that defines
  underrepresentation ([`docs/RESEARCH.md`](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting))
  does not move.
- Runs at targets 30 and 40 on the judged configuration: the baselines free; the BKO arms' gap
  fills paid at an estimate of the target over 20 times the gap fill's share of the judged run's
  spend, their replica fills answered from the judged run's log; each run detached, kept awake, on
  mains power and watched as `e1-3` was.
- Per target: each arm's measures, each contrast's corrected interval, validity, drops, privacy and
  test copies, in a new `docs/EVIDENCE.md` entry that reads the judged run as the target-20 point.

**Out.** Changing the threshold; a verdict (the criteria read the judged configuration at its
threshold).

## Acceptance

- Each target's measures regenerate from a committed command and its run is recorded with its
  spend.
- `make check`.
