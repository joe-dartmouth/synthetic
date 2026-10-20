---
id: proposal-e1-replica-run
title: E1 with BKO replicas
status: ready
depends_on: --
blocks: proposal-guidance-ablation
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# E1 with BKO replicas

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Every row-adding arm fills a replica (E1's Configuration in
[`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation)). This unit runs E1 once
in that configuration, so the BKO arms' fidelity and replica utility sit beside the baselines' from
one run in which every arm is treated the same.

## Scope

**In.**

- Before the run, the `live` generation test and one paid completeness request at the Haiku 5.5
  settings and its closure (C1's pair of commands,
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#gap-closure)), recorded as a C1 Runs row with its
  spend, its tokens per record asked and its validity (`passed` summed over `lines` summed, from
  `bko.fusion`'s `join` lines). It supersedes `c1-1`, as C1's Configuration rules for a change to
  what a call sends, so C1's Result and Read against are rewritten from its closure output. If its
  validity is below `E1.validity`, or a call fails, stop and report before the run: a low pass rate
  means top-ups and refused attempts, which the run's budget and retries may not absorb.
- A free rehearsal on all holdouts with canned replies that answer each call's own cell (a replica
  asks every cell), then one paid `score` run of the replica configuration under the recorded
  ceiling, then its replay and `compare`, as for `e1-1`. The spend and duration estimate is the one
  E1's Configuration in `docs/EVIDENCE.md` records.
- Run the paid command detached and kept awake (for example `caffeinate -i` with the guarded command
  in the background, output to `scratchpad/`). Check the process is alive, watch the `holdout`
  progress and `spend` lines, and stop to report if spend passes 80% of the ceiling before the last
  holdout, or if after the fifth holdout the spend times 20 exceeds the ceiling.
- Copy the call log off the runner after the run and before replay; the Runs row names where. The
  call log, replay log and stderr stay in gitignored `scratchpad/`; only their sha256s and counts
  are committed.
- A failed run stops the unit: record its Runs row with the reason and spend, and report to the
  user before any rerun.
- Record the run as E1's judged result in its `docs/EVIDENCE.md` entry: a new Runs row with spend
  and the replay check, Status `complete`, and the Result and Read against fields written from its
  `compare` (each verdict, validity with the gap and replica counts apart, every arm's replica
  fidelity and replica utility, and the test-copy counts). `e1-1` keeps its Runs row.

**Out.** Any code change (a needed one stops the run and
goes through a new PR).

## Acceptance

- Every number is regenerable from a committed command and recorded in `docs/EVIDENCE.md`, with
  the run's spend in its Runs field.
- `make check`.
