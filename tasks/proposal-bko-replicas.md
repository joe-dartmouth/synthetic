---
id: proposal-bko-replicas
title: BKO arms fill a replica, as every row-adding arm does
status: ready
depends_on: --
blocks: proposal-e1-replica-run
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# BKO arms fill a replica, as every row-adding arm does

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

In E1's judged run `e1-1` the four row-adding baselines fill a replica of the training split's
cells and the three BKO arms fill none, so the BKO arms have no fidelity or replica utility
reading beside the baselines'. The team chose (2026-10-19) to give every row-adding arm the same
treatment, accepting the replica spend. `e1-1`'s data cannot be reused for this: its call log
holds no replica call, replica rows enter the test-split decontamination of every arm, and
resuming a run by replaying its log and then calling live is ruled out (ADR
`research/call-log-replay`). So the replica run is a new paid E1 run, in
[`proposal-e1-replica-run`](proposal-e1-replica-run.md); this unit makes the code and readings
ready for it, with no spend.

## Scope

**In.**

- `bko`, `bko-population` and `bko-frequency` join `E1.replica_arms`, all three, as ADR
  `research/e1-evaluation-protocol` (Reversal) allows once their replica spend is recorded in
  E1's Configuration before a run.
- Each BKO arm fills `bko.baselines.request.replica_request` the way it fills the gap request:
  same guidance, rounds, budget, drops and refusal rule, and the same replica measures as the
  baselines (replica utility, fidelity, `test-copies`).
- Validity follows one rule for every candidate, as for every other measure: the rate counts
  every candidate `bko.validate` sees, gap and replica fills alike, as E1's Configuration already
  states; the gap and replica counts are also reported apart, with no separate verdict.
- Record in `docs/EVIDENCE.md` (E1's Configuration, or a new entry for the replica run) the
  replica spend estimate from `e1-1`'s cost per record asked, and a spend ceiling that covers it.
- Update RESEARCH section 5, the e1 ADR's Rationale, Ruled Out and Reversal, and
  `docs/DEVELOPMENT.md` to the new configuration; tests with fake sends only.
- Decide how the new run is recorded so `e1-1` stays E1's judged run: a later paid run is a new
  entry, never a replacement (ADR `research/e1-evaluation-protocol`, One judged run).

**Out.** Any paid call; changing `e1-1`'s record or verdicts.

## Acceptance

- `score` with the default experiment fills a replica for every row-adding arm, shown by a CLI
  test over scripted replies whose replay is byte-identical.
- The replica spend estimate and the ceiling are in `docs/EVIDENCE.md` before any paid run.
- `make check`.
