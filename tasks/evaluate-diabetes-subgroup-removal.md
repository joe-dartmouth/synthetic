---
id: evaluate-diabetes-subgroup-removal
title: Subgroup removal on Diabetes 130-US Hospitals
status: ready
depends_on: --
blocks: generate-bkb-curation-diabetes
parallel_safe: false
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# Subgroup removal on Diabetes 130-US Hospitals

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

UCI Heart Disease cannot separate the arms: its test splits are too small for the flagged
subgroups to move an interval, its thin cells carry no signal the training rows lack, and Sonnet
5.5 copied its test records word for word where Haiku 5.5 copied one
([`docs/RESEARCH.md`](../docs/RESEARCH.md#limits-of-uci-heart-disease-as-the-evaluation-frame)).
E3 shows the removal design on Synthea, whose patients are generated. This unit runs the same
design on real patients with enough subgroup records to read a difference: UCI Diabetes
130-US Hospitals, about a hundred thousand inpatient encounters with ICD-9 diagnoses and a
readmission label. Every arm and reading the plan holds is built, so it compares every method at
once.

## Scope

**In.**

- **Data.** The archive pinned in `bko.datasets.sources` by size and sha256 with its license and
  its `Agreement` read from the archive's terms on the day it is added; a frame
  (`bko.datasets.frames.FrameSpec`) with one encounter per patient, the readmission label, age,
  sex and race, the measured fields, and indicator columns for the diagnosis groups the ontology
  maps; and a mapping from those ICD-9 groups to MONDO and HPO concepts checked against the pinned
  releases, as the UCI and Synthea field rows are (`python -m bko.ontology check`). The CLLM
  arms send training records whole, so before any Messages arm runs the unit checks the archive's
  terms against [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#generation) (What leaves the
  machine).
- **Power first.** Before any run, a power calculation fixes the holdout count, the test split's
  size and the subgroup removed: the smallest effect on the subgroup's recall that the design can
  read, from the subgroup's test positives per holdout and E1's corrected interval, stated in the
  new ADR. The removed subgroup is clinically defined and holds enough positives that its test
  split can move an interval.
- **Design.** E3's removal rule on the real frame: a fraction of the subgroup's training positives
  removed (at least 0.9, and 1.0 as the zero-shot case where count-based methods have nothing to
  copy), the test split whole, and every arm the plan holds then (the comparison set, the
  classical synthesizers, the Bayesian network, `bko`, the CLLM arms and the BKB arms), with the
  reference arm `real` adding back the removed rows.
- **Readings.** Recall and AUC-PR on the removed subgroup and over the whole test split, each
  contrast's corrected interval, test copies per arm (the generator's memorization of the
  archive), and validity and drops, in a new `docs/EVIDENCE.md` entry; spend estimated before
  each run from the judged runs' tokens per record asked, under its ceiling and the paid-run
  protocol (`docs/DEVELOPMENT.md`, Evaluation).

**Out.** Credentialed data (MIMIC-IV); NHANES, the other open candidate, which needs survey
weights and a derived outcome; a change to E1's criteria or to UCI's runs.

## Acceptance

- The archive verifies against its pin, the frame and the ICD-9 mapping are tested on a fixture
  slice, and the mapping checks against the pinned releases.
- The power calculation and the chosen design are recorded in a new ADR before the first paid
  run.
- Each run regenerates from `score <run>` and `compare <run>`, is recorded with its spend, and
  replays byte for byte.
- `make check`.
