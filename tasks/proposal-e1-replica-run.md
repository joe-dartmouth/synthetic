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
  `e1-1`. Each BKO arm fills a 238-row replica per holdout beside its 20-row gap fill; at
  `e1-1`'s cost per record asked the run is about $245 and 17-20 h of calls.
- Record the run as E1's judged result in its `docs/EVIDENCE.md` entry: a new Runs row with spend
  and the replay check, and the Result and Read against fields rewritten from its `compare` (each verdict,
  validity with the gap and replica counts apart, every arm's replica fidelity and replica
  utility, and the test-copy counts). `e1-1` keeps its Runs row.

**Out.** Removing `e1-1`'s Runs row; any code change (a needed one stops the run and
goes through a new PR).

## Acceptance

- Every number is regenerable from a committed command and recorded in `docs/EVIDENCE.md`, with
  the run's spend in its Runs field.
- `make check`.
