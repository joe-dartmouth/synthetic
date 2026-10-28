---
id: evaluate-synthea-frame
title: The Synthea frame and E3's rules
status: ready
depends_on: --
blocks: evaluate-synthea-downstream
parallel_safe: false
owner: Joe
milestone: final-report
paid: false
adrs: research/synthea-downstream-design
---

# The Synthea frame and E3's rules

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E3 needs a Synthea frame and rules E1 does not have. The frame's rows are mostly duplicates, the
removed cell's ratio would widen every interval, and E3 needs an un-ablated reference inside the
run. `score` already reads each named run's frame from one `bko.evaluate.data.Data` value, so
this unit adds a dataset and E3's rules, and `evaluate-synthea-downstream` spends the money.

## Scope

**In.**

- The frame builder over the pinned Synthea population: one row per patient alive on the
  snapshot date and aged 40 and over, with age, sex, the conditions
  `bko.ontology.mapping.CODE_MAPPINGS` maps as indicators, and the label ischemic heart disease by
  the snapshot; no feature stands in for the label.
- The training-only removal: `bko.datasets.ablation`'s fraction draw applied to the training rows
  only, so the test split keeps every case.
- The un-ablated held-out ratio, so the removed cell's ratio does not widen every interval.
- Test copies counted, not removed, since the frame's rows are mostly duplicates.
- The `real` reference arm: the un-ablated training rows inside the run.
- The flagged-cell fingerprint per holdout, so runs that flag different cells do not pair.
- The named runs `e3`, `e3-a` and `e3-b` in `bko.evaluate.experiment.RUNS`, and a `merge`
  command that joins the slices' scores.
- A new ADR under `docs/adr/research/` fixing the fields, the label, the age bands, the split, the
  removed share, the copy policy and the run's ceiling, with the reason for each.
- ADRs [`research/record-generation-design`](../docs/adr/research/record-generation-design.md)
  and [`research/record-validation-design`](../docs/adr/research/record-validation-design.md)
  edited as their reversal conditions for a Synthea request say.

**Out.** Any paid run.

## Acceptance

- The frame's sha256 is equal across two separate-process builds.
- A `tmp_path` test builds the frame.
- `launch e3-1 e3 --rehearse` completes for free, with its go/no-go reading.
- `make check`.
