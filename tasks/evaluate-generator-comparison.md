---
id: evaluate-generator-comparison
title: The generator model at a fixed level
status: blocked
depends_on: proposal-guidance-ablation
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
model is the one generator setting no run has varied alone.

## Scope

**In.**

- `bko.generate.spend.PRICES` gains Sonnet 5.5's and Opus 5.5's rates, read from the pricing page
  `PRICES`' docstring names on the day they are added.
- The `bko` arm at the level the ablation picked, gap and replica fills on every holdout, with
  `SETTINGS.model` at Sonnet 5.5, then at Opus 5.5; each run's spend estimated before it from
  `e1-3`'s tokens per record asked at the model's rates, and the Opus run gated on an explicit
  sign-off against `E1.spend_ceiling`, since its rates are many times Haiku's; if the estimate
  passes the ceiling, the Opus run drops the replica fill or takes a fixed subset of holdouts, and
  its Runs row says which. Each run detached, kept awake, on mains power and watched as `e1-3` was.
- Per model, beside Haiku's judged figures: validity, drops, tokens and seconds per record asked,
  replica fidelity and utility, each criterion's measures with no verdict, in a new
  `docs/EVIDENCE.md` entry.

**Out.** Changing `SETTINGS.model` for the judged run (a result that argues for it goes through
[`docs/RESEARCH.md`](../docs/RESEARCH.md) and a new E1 configuration).

## Acceptance

- Each model's run is recorded with its spend and replays byte for byte.
- `make check`.
