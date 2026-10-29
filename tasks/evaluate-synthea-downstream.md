---
id: evaluate-synthea-downstream
title: A downstream experiment on Synthea with known gaps
status: blocked
depends_on: evaluate-synthea-frame
blocks: evaluate-classical-baselines, evaluate-bayesian-network-baseline, evaluate-cllm-baseline
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# A downstream experiment on Synthea with known gaps

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

UCI gives E1 two underrepresented cells and about five positive test records in the flagged
subgroups per holdout, and no arrangement of the archive adds more (ADR
[`research/uci-evaluation-frame`](../docs/adr/research/uci-evaluation-frame.md)). The pinned
Synthea population has cells as large as the ablation wants and a truth it fixes: remove a share
of the women with ischemic heart disease from the training rows, and the recall each method
recovers on the held-out cases is the measure of the augmentation itself. The frame, its rules and
the named runs come from `evaluate-synthea-frame`; this unit spends the money and records E3.

## Scope

**In.**

- The paid E3 run as its two slices, the named runs `e3-a` and `e3-b`, each launched under the
  paid-run protocol (`docs/DEVELOPMENT.md`, Evaluation), with its spend estimated before it runs
  from the judged run's (`e1-4`'s) tokens per record asked, 300.535 input and 211.162 output, and
  the frame's quotas, stated in the Runs row, under the run's ceiling.
- The `merge` of the two slices' scores into E3's scores, and `compare e3` over them.
- A new `docs/EVIDENCE.md` entry, E3, with its Configuration, Runs, Result and Read against: each
  method's measures, with recall on the removed cell's held-out cases as the headline, read
  against the un-ablated `real` reference arm.
- [`docs/RESEARCH.md`](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting) gaining the
  experiment and what it is read against.

**Out.** Replacing E1 as the judged experiment; credentialed data; any change to the frame or to
E3's rules.

## Acceptance

- Each slice replays byte for byte.
- The merged scores are recorded in E3 with its spend.
- `make check`.
