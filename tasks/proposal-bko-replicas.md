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
reading beside the baselines'. Every row-adding arm gets the same treatment, and the replica spend
is accepted. `e1-1`'s data cannot be reused for this: its call log holds no replica call, replica
rows enter the test-split decontamination of every arm, and resuming a run by replaying its log and
then calling live is ruled out (ADR `research/call-log-replay`). So the replica run is a new paid
E1 run, in [`proposal-e1-replica-run`](proposal-e1-replica-run.md); this unit makes the code and
readings ready for it, with no spend. `e1-1` keeps its Runs row.

## Scope

**In.**

- `bko`, `bko-population` and `bko-frequency` join `E1.replica_arms`, all three, as ADR
  `research/e1-evaluation-protocol` (Reversal) allows once their replica spend is recorded in
  E1's Configuration before a run. `E1.replica_arms` is defined in code from the row-adding arms,
  so the BKO arms must be row-adding there; `tests/test_evaluate_experiment.py`, which pins
  `replica_arms` to `Method.adds_rows` and `spend_ceiling` to 50.0, is updated.
- Each BKO arm fills `bko.evaluate.experiment.replica_request` the way it fills the gap request:
  same guidance, rounds, budget, drops and refusal rule, and the same replica measures as the
  baselines (replica utility, fidelity, `test-copies`).
- Validity follows one rule for every candidate: the rate counts every candidate `bko.validate`
  sees, gap and replica fills alike. With the arms in `replica_arms` that already holds, one
  `Tally` per arm. Reporting the gap and replica counts apart needs the tally and `drain` to split
  per fill and `compare` to log both; there is no separate verdict.
- Generator: `bko.generate.messages.SETTINGS` moves from `claude-opus-5-5` to `claude-haiku-5-5`,
  keeping effort `medium` (Haiku 5.5's default; it accepts the `effort` parameter), `max_tokens`,
  `per_call` and level `ontology`. `bko.generate.spend.PRICES` gains Haiku 5.5's rates, read on
  the day from `platform.claude.com/docs/en/about-claude/pricing.md` (Model pricing, the row for
  prompts up to 100,000 tokens); the Opus 5.5 entry stays so `e1-1`'s log still prices. The
  generator is a configuration value, so this is part of the replica configuration (ADR
  `research/e1-evaluation-protocol`, The latest configuration is judged). Update ADR
  `research/record-generation-design` (The call, the price), E1's and G1's Configuration in
  `docs/EVIDENCE.md`, and the tests that name the model where they mean the current setting.
- Spend: estimate the replica spend from `e1-1`'s tokens per record asked (its Runs row) priced at
  Haiku 5.5's rates, with a stated margin of at least 25%, set `E1.spend_ceiling` in code to cover estimate plus margin, and
  record the estimate, the margin and the ceiling in E1's Configuration in `docs/EVIDENCE.md`.
- Outage tolerance for a run of many hours: raise the SDK client's `max_retries` and timeout in
  `bko.generate.api.default_send`. This changes no logged request, so replay holds. Update the e1
  ADR's Rationale sentence on retries and its Ruled Out row on a call with no response as needed.
- Update RESEARCH section 5 and E1's Configuration in `docs/EVIDENCE.md` (both say the BKO arms
  fill none), the e1 ADR's Rationale, Ruled Out and Reversal (its Rationale and Ruled Out row on
  BKO arms in `E1.replica_arms` are rebutted there: every row-adding arm gets the same readings,
  and validity is deliberately read over every candidate, replica candidates included), and
  `docs/DEVELOPMENT.md`, to the new configuration; tests with fake sends only.
- The replica configuration's run becomes E1's judged result once it lands (ADR
  `research/e1-evaluation-protocol`, The latest configuration is judged).
- A paid generation check is left to `proposal-e1-replica-run`; this unit spends nothing.
- A CLI test in `tests/test_evaluate_bko_cli.py` over a fixture experiment with BKO replica arms
  (not the 100-holdout default): its scripted-replies helper is extended for replica calls, and a
  live run then its replay are byte-identical.

**Out.** Any paid call; changing `e1-1`'s Runs row.

## Acceptance

- The CLI test above passes: every row-adding arm fills a replica and the replay is
  byte-identical.
- `compare` logs the gap and replica candidate counts apart, shown by a test.
- `E1.spend_ceiling` is at least the estimate times 1.25, and the estimate, margin and ceiling are
  in E1's Configuration in `docs/EVIDENCE.md` before any paid run.
- A test shows `default_send` builds a client with the raised `max_retries` and timeout.
- `SETTINGS` names `claude-haiku-5-5`, and `PRICES` prices it (a test pins its rates to the
  pricing page's values recorded in the code).
- The docs listed above match the new configuration.
- `make check`.
