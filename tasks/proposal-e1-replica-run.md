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
This unit runs E1 five times in that configuration, so the BKO arms' fidelity and replica utility sit
beside the baselines' from one run in which every arm is treated the same.

## Scope

**In.**

- A free rehearsal on all holdouts with canned replies, then five paid `score` runs of the
  replica configuration with different seeds, each under the recorded ceiling, each replayed,
  then one combined `compare`. At `e1-1`'s cost per record asked, each run is about $245 and
  17-20 h of calls, so five are about $1,225 and 85-100 h; confirm the spend before the first.
- Record the five runs as E1's judged result in its `docs/EVIDENCE.md` entry: a Runs row per run
  with spend and replay check, and the Result and Read against fields rewritten from the combined
  `compare` (each verdict,
  validity with the gap and replica counts apart, every arm's replica fidelity and replica
  utility, and the test-copy counts). `e1-1` keeps its Runs row.

**Out.** Removing `e1-1`'s Runs row; any code change (a needed one stops the run and
goes through a new PR).

## Acceptance

- Every number is regenerable from a committed command and recorded in `docs/EVIDENCE.md`, with
  the run's spend in its Runs field.
- `make check`.
