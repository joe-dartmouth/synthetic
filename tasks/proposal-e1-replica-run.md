---
id: proposal-e1-replica-run
title: E1 with BKO replicas
status: blocked
depends_on: proposal-bko-replicas
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# E1 with BKO replicas

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

[`proposal-bko-replicas`](proposal-bko-replicas.md) makes every row-adding arm fill a replica.
This unit runs E1 once in that configuration, so the BKO arms' fidelity and replica utility sit
beside the baselines' from one run in which every arm is treated the same.

## Scope

**In.**

- A free rehearsal on all holdouts with canned replies, then one paid `score` run of the
  replica configuration under the recorded ceiling, then its replay and `compare`, as for
  `e1-1`. The spend and duration estimate is the one E1's Configuration in `docs/EVIDENCE.md`
  records.
- Run the paid command detached and kept awake (for example `caffeinate -i` with the guarded
  command in the background, output to `scratchpad/`). Check the process is alive, watch the
  `holdout` progress and `spend` lines, and stop to report if spend passes 80% of the ceiling
  before the last holdout.
- Copy the call log off the runner after the run and before replay; the Runs row names where. The
  call log, replay log and stderr stay in gitignored `scratchpad/`; only their sha256s and counts
  are committed.
- A failed run stops the unit: record its Runs row with the reason and spend, and report to the
  user before any rerun.
- Record the run as E1's judged result in its `docs/EVIDENCE.md` entry: a new Runs row with spend
  and the replay check, and the Result and Read against fields rewritten from its `compare` (each
  verdict, validity with the gap and replica counts apart, every arm's replica fidelity and
  replica utility, and the test-copy counts). `e1-1` keeps its Runs row.

**Out.** Any code change (a needed one stops the run and
goes through a new PR).

## Acceptance

- Every number is regenerable from a committed command and recorded in `docs/EVIDENCE.md`, with
  the run's spend in its Runs field.
- `make check`.
