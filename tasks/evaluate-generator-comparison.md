---
id: evaluate-generator-comparison
title: The generator model at a fixed level
status: blocked
depends_on: evaluate-e1-second-configuration, evaluate-named-runs
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
[`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md) fixes.

## Scope

**In.**

- `bko.generate.spend.PRICES` gains Sonnet 5.5's and Opus 5.5's rates, read from the pricing page
  `PRICES`' docstring names on the day they are added, with `tests/test_generate_spend.py` pinning
  the three rows.
- Two named runs (`evaluate-named-runs`) with the baselines and the `bko` arm at the judged
  configuration's level, bounds and effort, `SETTINGS.model` at Sonnet 5.5 and then at Opus 5.5,
  each run's spend estimated before it from the judged run's tokens per record asked at the
  model's rates. The Opus estimate at `e1-3`'s tokens passes `E1.spend_ceiling` many times over,
  so the Opus run has its own ceiling in its `Experiment`, and the commit that sets it is the
  sign-off; until that commit the unit records Sonnet alone. Each run under the protocol of
  `evaluate-run-protocol`.
- Per model, beside the judged Haiku figures at the same holdouts: validity, drops, tokens and
  seconds per record asked, replica fidelity and utility, each criterion's measures with no
  verdict, in a new `docs/EVIDENCE.md` entry.

**Out.** Changing `SETTINGS.model` for the judged run (a result that argues for it goes through
[`docs/RESEARCH.md`](../docs/RESEARCH.md) and a new E1 configuration).

## Acceptance

- Each model's run is recorded with its spend and replays byte for byte, or the entry states that
  the Opus run waits on its ceiling's commit.
- `make check`.
