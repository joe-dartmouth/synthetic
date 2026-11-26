---
id: datasets-nvd-frame
title: Add NVD CVE records with the KEV label as an exploratory non-health frame over CWE
status: blocked
depends_on: datasets-generalised-cell-axes
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/nvd-frame
---

# Add NVD CVE records with the KEV label as an exploratory non-health frame over CWE

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

ADR
[`research/non-health-dataset-selection`](../docs/adr/research/non-health-dataset-selection.md)
chooses NVD CVE records with the CISA KEV label, over the CWE hierarchy, as an exploratory
non-health family. Its record is a vulnerability, not a person, so the frame names its own cell
axes, which unit [`datasets-generalised-cell-axes`](datasets-generalised-cell-axes.md) makes
possible. The CVE API reported 432,554 records, 1,805 of them in KEV, on 2026-11-25 local
(2026-11-26 UTC; `services.nvd.nist.gov/rest/json/cves/2.0` with `resultsPerPage=1` and with
`hasKev`), so the frame fixes its snapshot before any gap is read.

## Scope

**In.**

- A pinned snapshot of the NVD CVE API 2.0's records, with its sha256, and the frame over it, as
  that ADR records:
  - the label: KEV membership, 1 when `cisaExploitAdd` is present on the record and 0 when it is
    absent;
  - the cells: `baseSeverity` by `attackVector` of the record's `cvssMetricV31` entry whose
    `type` is `Primary`, and the rule for a record with no such entry;
  - each record's CWE ids from `weaknesses` mapped to a pinned CWE release, converted from its
    XML so the ontology loader reads the ChildOf hierarchy;
  - any sampling of the snapshot, fixed before any gap is read;
  - a `python -m bko.gaps nvd` subcommand.
- The `Source` pin carries `agreement=None`, and the frame shows the NVD's attribution notice
  wherever it displays the data. Generated rows are modified content, so nothing attributes them
  to the NVD.
- A new ADR under `docs/adr/research/` that records the frame, the CWE release and the snapshot.

**Out.**

- The test split, any classifier fit or arm score, and picking the removed cell.
- Any other non-health candidate.

## Acceptance

- The unit's ADR records the frame's record count, counted from the training split before any gap
  is read. Below 5,000 records the dataset is excluded and ADR
  `research/non-health-dataset-selection` revised.
- `python -m bko.ontology check` passes with the new rows.
- `python -m bko.gaps nvd` lists the training split's gaps.
- `make check` passes.
