---
id: privacy-identifier-perturbation
title: Perturb identifying fields with an LLM, keeping concepts and distribution, and track distance per method
status: blocked
depends_on: evaluate-diabetes-subgroup-removal
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: research/identifier-perturbation
---

# Perturb identifying fields with an LLM, keeping concepts and distribution, and track distance per method

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

On Diabetes 130 with the target cell wholly removed, `bko-direct` is the strongest arm, but it
builds each record from blocks of real training records' field values
(`bko.generate.direct.sample`): every field group comes from one training record showing exactly
the drawn concepts, and every free field from a record of the cell or its label group. Its added
rows show no whole-record test copy, and their 5th-percentile distance to the closest training
record sits beside CTGAN's, but a block of fields is still one real patient's. The fields a
patient could be found by in this frame are the ones no field row maps to a concept: `race`, the
eight count fields (`time_in_hospital`, `num_lab_procedures`, `num_procedures`,
`num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`,
`number_diagnoses`), the eight drug columns, `change` and `diabetesMed`. A pass that rewrites
those fields while leaving every mapped concept, the cell coordinates (`age`, `sex`, the label)
and the column distributions in place would move each record away from its source patients
without changing what the arm teaches a classifier. The privacy criterion
([`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria)) reads distance to closest record,
so the question is measured per method on the same holdouts.

## Scope

**In.**

- A perturbation step over an arm's added rows. A Messages call at `SETTINGS`' model receives a
  batch of rows and rewrites only the unmapped fields above, within each field's training range
  and coded states, and returns them in the generation format. Each returned row is refused, and
  the original kept and counted, when any of these differ from the input:
  - a mapped concept's state, read through the frame's field rows;
  - `age`, `sex` or the label;
  - a coded field's state set.
- A distribution check per holdout. SDMetrics column shapes and column pair trends compare the
  perturbed rows with the unperturbed rows. A refusal rule, fixed in the new ADR before any run,
  sets the score below which the perturbation is refused for that holdout.
- A named run in `bko.evaluate.named_runs.RUNS` on E6's Diabetes frame and removal at fraction
  1.0. Its arms are `bko-direct` and the perturbed arm `bko-direct-perturbed`, with `none` and
  `real` as references. Its spend is estimated in the Runs row before the run, under its ceiling
  and the paid-run protocol ([`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation)).
- A per-method privacy table for E6, at both fractions, from the existing scores. It holds every
  row-adding arm's added rows' `dcr-p5` and `dcr-median`, the held-out rows' as the baseline, and
  `test-copies`. The perturbed arm sits beside them, with its paired difference against
  `bko-direct` on recall, AUC-PR and both distances, each with its corrected interval.
- A new ADR under `docs/adr/research/` recording the fields perturbed, the refusal rules and the
  distribution threshold, with their reasons, and a new `docs/EVIDENCE.md` entry for the result.

**Out.**
- Formal privacy guarantees: differential privacy and k-anonymity proofs.
- Membership-inference attacks.
- Perturbing arms other than `bko-direct`.
- Any change to an existing arm's calls. E6's logs must stay replayable.

## Acceptance

- Each perturbed row keeps every mapped concept, the cell coordinates and the coded state sets of
  its input, and a test plants a row that changes each one and sees it refused.
- The privacy table holds every row-adding arm of both E6 fractions, read from the merged scores
  with no API call.
- The perturbed run is recorded with its spend, replays byte for byte, and reports the paired
  differences against `bko-direct` on utility and distance.
- `make check`.
