---
id: privacy-identifier-perturbation
title: Perturb identifying fields with an LLM, keeping concepts and distribution, and track distance per method
status: ready
depends_on: --
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

## Findings so far

The arms, the check and the pilots exist (ADR
[`research/identifier-perturbation`](../docs/adr/research/identifier-perturbation.md)). Every
Diabetes field group is a single field, and `bko-direct` picks each free field from its own
record, so its rows are already mosaics: their 19-field combination matches a training record for
under 1% of rows, below `real`'s rate. The pilots on holdouts 0 and 1
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e10-identifier-perturbation)) cost about 0.00019 USD per
row asked. The rewrite returned every row asked, but the distribution check refused
`bko-direct-perturbed` on 3 of 4 holdouts, both at the empty fraction, keeping the original rows;
PRAM passed on all four. Read descriptively on two holdouts, both arms lowered membership-inference
AUC in the removed cell at the thin fraction and moved the added rows' 5th-percentile distance
away from the training records, with recall and AUC-PR within a few points of `bko-direct`. The
ADR's reversal condition on a check that refuses every holdout of a fraction is met for the
rewrite arm at the empty fraction, so its reference is reopened before a full run. A full run of
both arms at both fractions over E6's 100 holdouts is estimated in E10's Configuration.

## Scope

**In.**

- Settle the rewrite arm's check reference under the ADR's reversal condition, recorded in the ADR
  before any full run.
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
- Any change to an existing arm's calls. E6's logs must stay replayable.

## Acceptance

- The full runs are recorded with their spend and replay byte for byte.
- E10's Result reports both arms' paired differences against `bko-direct` on utility, distance
  and membership inference at both fractions, read from the stored rows with no API call.
- `make check`.
