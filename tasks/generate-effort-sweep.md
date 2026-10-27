---
id: generate-effort-sweep
title: The lowest effort that keeps validity and fidelity
status: blocked
depends_on: evaluate-second-classifier, proposal-guidance-ablation, evaluate-frame-generic-pipeline
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

- A named run (`bko.evaluate.experiment.RUNS`) over the first 10 holdout seeds with the baselines and the
  `bko` arm, one per effort, `low` and `medium`, at `SETTINGS` otherwise unchanged; both run live,
  since `e1-3`'s log interleaves three arms over 100 holdouts and replays only whole. Spend
  estimate: for each effort, a tenth of one arm's share of `e1-3`'s spend, `low`'s below it.
- Per effort, paired over the same 10 holdouts: validity, candidates dropped as outside the
  range, replica fidelity (quality, column shapes, column pair trends), replica utility, tokens per
  record asked, seconds per call, and spend.
- The decision, stated here before the run: `SETTINGS.effort` moves to the lowest effort whose
  validity rate meets `E1.validity` and whose replica fidelity is not below `medium`'s on any of
  the three measures over the paired holdouts, or stays `medium`. The move changes what every
  call sends, so it is a new configuration of E1 that
  [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md) runs.
- A new `docs/EVIDENCE.md` entry for the sweep, with each run's spend and folder in its Runs field
  and the measure that decided the effort in its Result.

**Out.** A full E1 run; any other change to `SETTINGS`.

## Acceptance

- Each effort's measures come from `score <run> --log` and `compare <run>` over its run folder,
  recorded in `docs/EVIDENCE.md` with the run's spend, and replay byte for byte.
- `SETTINGS.effort` is set by the stated rule.
- `make check`.
