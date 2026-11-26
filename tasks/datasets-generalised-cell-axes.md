---
id: datasets-generalised-cell-axes
title: Let a frame name its own two cell axes in place of age band and sex
status: ready
depends_on: --
blocks: datasets-nvd-frame, datasets-vcdb-frame
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: research/generalised-cell-axes
---

# Let a frame name its own two cell axes in place of age band and sex

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Every frame so far describes a person, so a cell is an age band by a sex. The non-health frames
ADR
[`research/non-health-dataset-selection`](../docs/adr/research/non-health-dataset-selection.md)
chooses, NVD CVE records with the KEV label and the VERIS Community Database, have other axes. The
cell machinery has to read them before any such frame loads.

## Scope

**In.**

- A frame names its own two cell axes and their bands in place of age band and sex.
- Cells, quotas, gap listings and the removed-cell rule read them.
- Every existing frame keeps age band by sex.
- Each non-health frame unit, [`datasets-nvd-frame`](datasets-nvd-frame.md) and
  [`datasets-vcdb-frame`](datasets-vcdb-frame.md), depends on this unit.
- A new ADR under `docs/adr/research/` that records the generalisation.

**Out.**

- Any new dataset.

**Findings from planning.**

- NVD and VCDB each need two coded axes.
- Axis names reach disk only in the gap listing's CSV headers and the Synthea ablation log, and
  every identity-checked file holds values, so the person frames' labels stay byte-identical.
- The Messages prompt, its reply schema and the rehearsal parser stay person-only.
- It reverses the Ruled Out row "A `cell_columns` field on the frame" of ADR
  [`research/uci-evaluation-frame`](../docs/adr/research/uci-evaluation-frame.md), with a dated
  revision of that ADR.
- `bko.gaps.quota.expected_size`, which unit `gaps-share-based-quota` adds, reads the second
  cell axis by position, which this unit keeps.

## Acceptance

- `DRAW_DIGEST` and `ADDED_DIGEST` are unchanged: `tests/test_generate_direct.py` pins the first
  and `tests/test_evaluate_direct.py` the second.
- A fixture frame with non-person axes fills a request.
- `rows` on an existing run folder reproduces its store: `rows e10-pilot-thin` over `runs/e10-3`
  writes a `rows.jsonl` that `cmp` finds equal to `runs/e10-3-rows/rows.jsonl`.
- `score --replay` of E1's judged paid run, `score e1 --replay runs/e1-5/calls.jsonl`
  ([E1](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation)), writes a `scores.csv` equal to
  `runs/e1-5/scores.csv`.
- `make check` passes.
