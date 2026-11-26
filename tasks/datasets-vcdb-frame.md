---
id: datasets-vcdb-frame
title: Add the VERIS Community Database as an exploratory non-health frame over the VERIS enumerations
status: blocked
depends_on: datasets-generalised-cell-axes
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/vcdb-frame
---

# Add the VERIS Community Database as an exploratory non-health frame over the VERIS enumerations

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

ADR
[`research/non-health-dataset-selection`](../docs/adr/research/non-health-dataset-selection.md)
chooses the VERIS Community Database as an exploratory non-health family. Its record is a security
incident, not a person, so the frame names its own cell axes, which unit
[`datasets-generalised-cell-axes`](datasets-generalised-cell-axes.md) makes possible. Its sample
mixes random and chosen incidents, and some records are encoded by AI, so the frame records both
shares.

## Scope

**In.**

- The repository's validated incidents at a pinned commit, with the sha256 of what is read, and
  the frame over them, as that ADR records:
  - the label: a breach, `attribute.confidentiality.data_disclosure` equal to `Yes`;
  - the cells: NAICS sector, the first two digits of `victim.industry`, by victim size from
    `victim.employee_count`, with the bands and the rule for an unknown value;
  - the VERIS action, variety and asset enumerations from the pinned schema, converted so the
    ontology loader reads them, and NAICS sectors over their codes;
  - the shares of non-random incidents (`plus.sub_source`) and AI-encoded incidents;
  - a `python -m bko.gaps vcdb` subcommand.
- The `Source` pin carries `agreement=None`. The ADR states that a published fill carries
  CC BY-SA 4.0 and VCDB's attribution.
- A new ADR under `docs/adr/research/` that records the frame, the schema release and the commit.

**Out.**

- The test split, any classifier fit or arm score, and picking the removed cell.
- Any other non-health candidate.

## Acceptance

- The unit's ADR records the frame's record count, counted from the training split before any gap
  is read. Below 5,000 records the dataset is excluded and ADR
  `research/non-health-dataset-selection` revised.
- `python -m bko.ontology check` passes with the new rows.
- `python -m bko.gaps vcdb` lists the training split's gaps.
- `make check` passes.
