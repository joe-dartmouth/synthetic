---
id: generate-effort-sweep
title: The lowest effort that keeps validity and fidelity
status: ready
depends_on: --
blocks: evaluate-e1-second-configuration
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# The lowest effort that keeps validity and fidelity

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

At effort `medium`, every one of `e1-3`'s 9340 replies holds a `thinking` block, against 13 of
603 in `e1-1`'s; output tokens per record asked are 197.046 against `e1-1`'s 94.163, and output is
about three quarters of the spend at `bko.generate.spend.PRICES` (E1's Runs rows,
[`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation); the block counts come
from `jq -r 'select(.kind=="reply") | .response.content[0].type'` over each call log). Whether a
lower effort keeps validity and replica fidelity is unmeasured, and the answer sets
`SETTINGS.effort` for every later paid run.

## Scope

**In.**

- Two named runs (`bko.evaluate.experiment.RUNS`), `g4-low` and `g4-medium`, over the first 100
  holdout seeds with the baselines and the `bko` arm, one per effort, at `summary`, the level G3
  chose and the one [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md)
  judges at; both run live, since a call log replays only whole. Their settings, launch commands
  and spend estimate are G4's Configuration
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g4-effort-at-summary)).
- Per effort, paired over the same 100 holdouts: validity, candidates dropped as outside the
  range, replica fidelity (quality, column shapes, column pair trends), replica utility, tokens per
  record asked, seconds per call, and spend.
- The decision, by the rule in G4's Configuration
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g4-effort-at-summary)), fixed before the runs: `low`
  or `medium`. A move to `low` changes what every call sends, so it is a new configuration of E1
  that [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md) runs.
- A new `docs/EVIDENCE.md` entry for the sweep ([G4](../docs/EVIDENCE.md#g4-effort-at-summary)),
  with each run's spend and folder in its Runs field and the measure that decided the effort in
  its Result.

**Out.** A full E1 run; any other change to `SETTINGS`.

## Acceptance

- Each effort's measures come from `score <run> --log` and `compare <run>` over its run folder,
  recorded in `docs/EVIDENCE.md` with the run's spend, and replay byte for byte.
- G4 records the effort the stated rule chooses, and `evaluate-e1-second-configuration` applies it
  to `SETTINGS.effort`.
- `make check`.
