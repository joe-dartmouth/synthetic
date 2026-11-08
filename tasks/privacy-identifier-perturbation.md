---
id: privacy-identifier-perturbation
title: Run the identifier-perturbation arms in full and read distance and membership inference per method
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: research/identifier-perturbation
---

# Run the identifier-perturbation arms in full and read distance and membership inference per method

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

## Pilot findings

The arms, the reading, the quotas and the pilots exist (ADR
[`research/identifier-perturbation`](../docs/adr/research/identifier-perturbation.md), which also
says why `bko-direct`'s rows are already mosaics). E10's Runs field
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e10-identifier-perturbation)) holds the pilots'
readings and Configuration the full-run estimate. What they settle for this unit: the rewrite
returns every row asked, but the distribution check refused it on thin holdout 1 and on both empty
holdouts, which meets the ADR's reversal condition for a check that refuses every holdout of a
fraction; PRAM passed on all four. Two holdouts read no direction, so the full runs decide both
arms' privacy and utility. The refusals came from the rewrite, which missed the label-1 shapes
that real data reaches, and a per-holdout gate biases the paired contrast, so a reading replaces
the check and quotas aim the rewrite (the ADR's Rationale).

## Scope

**In.**

- Replace the rewrite arm's check with a reading that never changes a row, and give each rewrite
  call per-call value quotas from the cell's label group, recorded in the ADR before any full run.
- Full runs of both arms at both fractions over E6's holdouts, as named runs in
  `bko.evaluate.named_runs.RUNS`, each with its spend estimated in the Runs row before the run,
  under its ceiling and the paid-run protocol
  ([`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation)).
- Their `rows`, `privacy` and `compare` readings: each arm's paired difference against
  `bko-direct` on recall, AUC-PR, distance to closest record and membership inference, each with
  its corrected interval, recorded as E10's Result.

**Out.**
- Formal privacy guarantees: differential privacy and k-anonymity proofs.
- Perturbing arms other than `bko-direct`.
- Any change to an E6 arm's calls; E6's logs must stay replayable.

## Acceptance

- The ADR records the reading and the quotas, and a paid re-pilot at them is recorded before any
  full run.
- Each full run's spend is estimated in E10's Runs row before it starts and stays under its
  ceiling.
- The full runs are recorded with their spend and replay byte for byte.
- E10's Result reports both arms' paired differences against `bko-direct` on utility, distance
  and membership inference at both fractions, read from the stored rows with no API call.
- `make check`.
