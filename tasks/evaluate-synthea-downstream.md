---
id: evaluate-synthea-downstream
title: A downstream experiment on Synthea with known gaps
status: blocked
depends_on: detect-synthea-mappings
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: research/synthea-downstream-design
---

# A downstream experiment on Synthea with known gaps

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

UCI gives E1 two underrepresented cells and about five positive test records in the flagged
subgroups per holdout, and no arrangement of the archive adds more (ADR
[`research/uci-evaluation-frame`](../docs/adr/research/uci-evaluation-frame.md)). The pinned
Synthea population has cells as large as the ablation wants and a truth it fixes: remove women with
ischemic heart disease from the training rows, and the recall each method recovers on the held-out
cases is the measure of the augmentation itself.

## Scope

**In.**

- A tabular frame from the pinned population: one row per patient alive on the snapshot date, with
  age, sex, the conditions [`detect-synthea-mappings`](detect-synthea-mappings.md) maps as
  indicators, and the label ischemic heart disease by the snapshot; a new ADR under
  `docs/adr/research/` fixes the fields, the label, the age bands, the split and the ablation it
  evaluates, with the reason for each.
- Cell-stratified holdouts as E1's, with the ablation applied to the training rows only: women
  with the condition removed to a fraction, so the test split keeps every case.
- Every arm of E1 on the frame: the baselines as they are, the BKO arms with a Synthea prompt
  (fields, codes, ranges) and guidance from the Synthea BKO; the measures E1 reports, with recall
  on the removed cell's held-out cases as the headline, read against the un-ablated training set.
  Spend estimated before the run from `e1-3`'s tokens per record asked and the frame's quotas,
  stated in the Runs row; the run detached, kept awake, on mains power and watched as `e1-3` was.
- A new `docs/EVIDENCE.md` entry, E3, with its Configuration, Runs, Result and Read against,
  [`docs/RESEARCH.md`](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting) gaining the
  experiment and what it is read against.

**Out.** Replacing E1 as the judged experiment; credentialed data.

## Acceptance

- The frame builds from the pinned population identically on a second run, pinned by its sha256 in
  the entry.
- The run is recorded with its spend and replays byte for byte.
- `make check`.
