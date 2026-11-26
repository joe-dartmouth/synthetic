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
  - the first cell axis: the victim's NAICS sector, the first two digits of `victim.industry`,
    over the Census sectors with 31-33, 44-45 and 48-49 each merged into one value; `00`, a
    missing code or an unparseable one is `unknown`;
  - an incident whose `victim.industry` holds several `;`-separated codes takes their sector when
    every code falls in one sector, and the value `multiple` when they span sectors, so each
    incident has one value;
  - the second cell axis: victim size from `victim.employee_count`, `small` for `Small`, `1 to 10`,
    `11 to 100` and `101 to 1000`, `large` for `Large` and every band from `1001 to 10000` up, and
    `unknown` for `Unknown` or a missing value, following VERIS's labels at 1,000 employees;
  - the VERIS action, variety and asset enumerations from the pinned schema, converted so the
    ontology loader reads them;
  - the shares of non-random incidents (`plus.sub_source`) and AI-encoded incidents;
  - a `python -m bko.gaps vcdb` subcommand.
- The `Source` pin carries `agreement=None`. VCDB's share-alike condition is taken to bind, as
  ADR `research/non-health-dataset-selection` records: any published fill derived from VCDB
  carries CC BY-SA 4.0 and VCDB's attribution, and the frame's ADR says so.
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
