---
id: proposal-e1-judged-run
title: E1's judged run
status: ready
depends_on: --
blocks: proposal-guidance-ablation
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# E1's judged run

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The Final Report milestone reads every criterion against the baselines and controls
([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria), [ROADMAP](../docs/ROADMAP.md#final-report)).
`bko.evaluate` holds E1's arms and readings; this unit runs E1 once and records it.

## Scope

**In.**

- Run E1 once as configured in `docs/EVIDENCE.md`, replay its call log, judge it with `compare`,
  and record:
  - the run and the replay check;
  - each verdict, with a null result reported as such;
  - `bko`'s validity rate beside its dropped candidates;
  - the ablation arms' rates and means, with no verdict;
  - the test-copy counts and the spend.
- Re-run E2's two free commands at the same commit and record whether their output still matches
  `e2-4`'s.

**Out.** Slides and the demo itself; any change to a reading.

## Acceptance

- Every number in the report is regenerable from a committed command and recorded in
  `docs/EVIDENCE.md`, with each run's spend in its Runs field.
- `make check`.
