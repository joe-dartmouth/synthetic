---
id: datasets-nhanes-frame
title: Add NHANES August 2021 to August 2023 as a confirmatory frame with a cardiovascular composite label
status: blocked
depends_on: datasets-selection-rule
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
general population. The composite outcome's size is what unit
[`datasets-selection-rule`](datasets-selection-rule.md) re-reads from the primary page.

## Scope

**In.**

- As unit [`datasets-support2-frame`](datasets-support2-frame.md), for NHANES August 2021 to
  August 2023:
  - pins of each component file, joined on the respondent sequence number, each carrying the
    `agreement` value unit [`datasets-selection-rule`](datasets-selection-rule.md) records, `None`
    only where that unit's ADR says the terms permit sending records to a third-party API;
  - adults only;
  - the label, a composite of the heart-failure, coronary-disease, heart-attack and stroke
    questions; its definition and the handling of "Don't know" answers are fixed in the new ADR
    under `docs/adr/research/` before any count is read;
  - age top-coded as published;
  - a `python -m bko.gaps nhanes` subcommand.
- Every reading runs on it, privacy included, on the reading unit
  [`datasets-selection-rule`](datasets-selection-rule.md) records.

**Out.**

- As unit [`datasets-support2-frame`](datasets-support2-frame.md): the test split, any classifier
  fit or arm score, and picking the removed cell.
- Survey weights.
- A diabetes label: glycated haemoglobin nearly defines it.

## Acceptance

- `python -m bko.ontology check` passes with the new rows.
- `python -m bko.gaps nhanes` lists the training split's gaps.
- The `contradictions` reading's output over the training split is recorded in the unit's ADR.
- `make check` passes.
