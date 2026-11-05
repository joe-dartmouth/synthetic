---
id: evaluate-membership-inference
title: Membership inference and NNDR for every row-adding arm, beside distance to closest record
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/membership-inference-reading
---

# Membership inference and NNDR for every row-adding arm, beside distance to closest record

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md) names three privacy measures: distance to closest record
(DCR), nearest-neighbour distance ratio (NNDR) and membership-inference risk. Only DCR is measured
(`bko.evaluate.privacy`).

E6 showed why DCR alone is not enough
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)):
- At f = 0.985, ADASYN's added rows' 5th-percentile DCR is 0.0000, and `bko-direct`'s is 0.0194,
  at equal removed-cell recall.
- `bko-direct` copies each field's value from a training record.

A low DCR says a row is near some training record. A membership-inference attack says whether the
rows let an attacker tell training records from held-out ones. Reviewers of synthetic health data
expect that second reading.

## Scope

**In.**

- **The attack.** A distance-based membership-inference attack over each arm's added rows. Every
  training record and every held-out record of the same holdout gets a score: its distance to the
  nearest added row, in the frame's columns. Report:
  - the attack's AUC (0.5 means no leak);
  - its true-positive rate at a low false-positive rate fixed in the ADR.

  The attack is read at both scopes, all records and the removed cell's. The cell is where E6's
  copying methods draw on the 19 kept records.
- **NNDR.** For each added row, the ratio of the distances to its nearest and second-nearest
  training records, with the same summaries DCR reports.
- **The rows.**
  - Rebuilt from stored files, at E6's run commit, for every arm whose rows a stored file holds:
    `bko-direct`, the Messages arms through `holdout-candidates.jsonl`, and `real`.
  - Regenerated locally from the named run's seeds for the classical and synthesizer arms, with no
    API call.

  Regeneration costs compute, so the unit may read a seeded subset of E6's holdouts that the ADR
  fixes before any run. Each regenerated arm line's sha256 must equal the run's.
- **Records.**
  - A new ADR under `docs/adr/` (`research/membership-inference-reading`) fixing the attack, the
    distance, the low false-positive rate, the holdout subset and the reading's decision rule.
  - A new `docs/EVIDENCE.md` entry for both E6 fractions.
  - A note in `tasks/privacy-identifier-perturbation.md` that its perturbed arm is read with the
    same attack.

**Out.**
- Shadow-model attacks.
- Differential-privacy training of any generator.
- Attribute-inference attacks.
- Any paid arm or API call.

## Acceptance

- Every row-adding arm of both E6 fractions has an attack AUC and its low-false-positive-rate
  true-positive rate at both scopes, and an NNDR summary. Each comes from a committed command that
  makes no API call.
- Every rebuilt or regenerated arm's rows match the run's logged sha256 on every holdout read.
- A test plants an arm that copies training records and sees an attack AUC well above 0.5, and one
  that draws independently of the training records and sees about 0.5.
- `make check`.
