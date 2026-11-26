---
id: datasets-generalised-cell-axes
title: Let a frame name its own two cell axes in place of age band and sex
status: blocked
depends_on: datasets-non-health-selection
blocks: datasets-nvd-frame, datasets-vcdb-frame
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/generalised-cell-axes
---

# Let a frame name its own two cell axes in place of age band and sex

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Every frame so far describes a person, so a cell is an age band by a sex. A chosen non-health
frame by unit [`datasets-non-health-selection`](datasets-non-health-selection.md) has other
axes. The cell machinery has to read them before any such frame loads.

## Scope

**In.**

- A frame names its own two cell axes and their bands in place of age band and sex.
- Cells, quotas, gap listings and the removed-cell rule read them.
- Every existing frame keeps age band by sex.
- Each non-health frame unit [`datasets-non-health-selection`](datasets-non-health-selection.md)
  adds depends on this unit.
- A new ADR under `docs/adr/research/` that records the generalisation.

**Out.**

- Any new dataset.

## Acceptance

- `DRAW_DIGEST` and `ADDED_DIGEST` are unchanged: `tests/test_generate_direct.py` pins the first
  and `tests/test_evaluate_direct.py` the second.
- A fixture frame with non-person axes fills a request.
- `make check` passes.
