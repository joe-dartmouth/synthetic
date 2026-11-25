---
id: datasets-nhanes-frame
title: Add NHANES August 2021 to August 2023 as a confirmatory frame with a cardiovascular composite label
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/nhanes-frame
---

# Add NHANES August 2021 to August 2023 as a confirmatory frame with a cardiovascular composite label

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

NHANES is the one candidate whose laboratory values map through cited clinical cutoffs in a
general population. ADR
[`research/confirmatory-dataset-selection`](../docs/adr/research/confirmatory-dataset-selection.md)
records the bounds the codebook's component counts set on the composite outcome's size; this unit
computes the union.

## Scope

**In.**

- As ADR [`research/support2-frame`](../docs/adr/research/support2-frame.md) records, for NHANES August 2021 to
  August 2023:
  - pins of each component file, joined on the respondent sequence number, each carrying the
    `agreement` value ADR `research/confirmatory-dataset-selection` records;
  - adults only;
  - the label, a composite of the heart-failure, coronary-disease, heart-attack and stroke
    questions; its definition and the handling of "Don't know" answers are fixed in the new ADR
    under `docs/adr/research/` before any count is read;
  - age top-coded as published;
  - a `python -m bko.gaps nhanes` subcommand.
- A new ADR under `docs/adr/research/` that records the frame and the mapping.
- Every reading runs on it, privacy included, on the reading of the NCHS terms ADR
  `research/confirmatory-dataset-selection` records.

**Out.**

- As ADR [`research/support2-frame`](../docs/adr/research/support2-frame.md) records: the test split, any classifier
  fit or arm score, and picking the removed cell.
- Survey weights.
- A diabetes label: glycated haemoglobin nearly defines it.

## Acceptance

- The unit's ADR records the adult frame's record count and the composite outcome's size, both
  counted from the training split before any gap is read. Below 5,000 records the dataset is
  excluded and ADR `research/confirmatory-dataset-selection` revised before registration.
- `python -m bko.ontology check` passes with the new rows.
- `python -m bko.gaps nhanes` lists the training split's gaps.
- The `contradictions` reading's output over the training split is recorded in the unit's ADR.
- `make check` passes.
