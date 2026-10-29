---
id: evaluate-generator-comparison
title: The generator model at a fixed level
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# The generator model at a fixed level

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

`e1-1` generated with Opus 5.5 and no replica, `e1-3` with Haiku 5.5 and a replica, so no
difference between them is attributable to the model
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation), Read against). The
model is the one generator setting no run has varied alone, and it can be varied alone only
against the judged run's level, bounds and effort, which
[E1](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation) fixes at `e1-4`.

## Scope

**In.**

- `bko.generate.spend.PRICES` gains Sonnet 5.5's rates, read from the pricing page `PRICES`'
  docstring names on the day they are added, with `tests/test_generate_spend.py` pinning the rows.
- Four named runs (`bko.evaluate.named_runs.RUNS`), each the baselines and the `bko` arm at the
  judged configuration's level, bounds and effort with `SETTINGS.model` alone changed: Sonnet 5.5
  through the Messages API (`g6-sonnet`), and three open models on this machine with thinking off
  (`g6-gemma`, `g6-nemotron`, `g6-olmo`; ADR `research/local-generator-backend`). Sonnet's spend is
  estimated before its run from the judged run's tokens per record asked at its rates; each run
  goes under the paid-run protocol (`docs/DEVELOPMENT.md`, Evaluation).
- Per model, beside the judged Haiku figures at the same holdouts: validity, drops, tokens and
  seconds per record asked, replica fidelity and utility, each criterion's measures with no
  verdict, in a new `docs/EVIDENCE.md` entry.

**Out.** Opus 5.5 (unscheduled, `docs/IDEAS.md`); thinking on for the local models (unit
`evaluate-local-thinking-cost`); changing `SETTINGS.model` for the judged run (a result that argues
for it goes through [`docs/RESEARCH.md`](../docs/RESEARCH.md) and a new E1 configuration).

## Acceptance

- Each model's run is recorded with its spend and replays byte for byte, or the entry states that
  a local model was not run under G6's duration rule (`docs/EVIDENCE.md`, G6, Configuration).
- `make check`.
