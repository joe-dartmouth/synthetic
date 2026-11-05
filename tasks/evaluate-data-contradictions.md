---
id: evaluate-data-contradictions
title: Find where the data contradicts the ontology, and show every stage handles it
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/data-contradiction-handling
---

# Find where the data contradicts the ontology, and show every stage handles it

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The instructor's feedback on the 2026-11-04 status update asked the team to look for
contradictions in the BKO that arise from the data, and to make sure the system handles those
edge cases.

R1 shows fusion keeping two conflicting ontologies side by side
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#r1-mondo-and-do-fusion-reproduction)). What no test or
reading covers is a dataset fragment that disagrees with the ontology fragments it is fused with.
Some cases:
- a record shows a narrow concept but not the broad one an `is_a` edge says it implies;
- a cell's share of a child concept exceeds its share of the parent;
- a cell has no records, or every record shows a concept;
- a field mapping's codes overlap another's.

Each one reaches gap detection (`bko.gaps`), the fused BKB's shares (`bko.gaps.shares`) and
`bko-direct`'s draw (`bko.generate.direct`), and none is tested there.

## Scope

**In.**

- **Phase 1, census.** A committed command that counts each contradiction kind in the UCI, Synthea
  and Diabetes 130 frames' training rows, per cell, reading the pinned releases. It makes no API
  call.
- **Phase 2, edge-case tests.** For each kind, a fixture in `tmp_path` that plants it, and tests
  that show what each stage does with it:
  - fusion: the fused BKB stays valid and keeps both the ontology's and the data's S-nodes;
  - gap detection: each definition's gaps, stated;
  - the shares: no share is negative or above 1, and a child is never drawn without its parent
    when the edge is nested;
  - `bko-direct`: a draw that no training record matches refuses with `SamplingError`, or falls
    back, as the ADR fixes.

  A stage that mishandles a case is fixed in this unit, or filed under a `cleanup:` key if the
  fix is outside it.
- **Phase 3, records.**
  - A new ADR under `docs/adr/research/` fixing what each stage does with each kind, and why.
  - A new `docs/EVIDENCE.md` entry for the census.

**Out.**
- Changing an ontology release or the field mappings to remove a contradiction. The unit reports
  and handles them; it does not edit the sources.
- Contradictions between ontologies, which R1 covers.

## Acceptance

- The census comes from a committed command over the pinned releases and the three frames, with
  no API call, and lists each kind's count per frame, or an empty list.
- Every kind has a planted test at each stage named above, and each passes.
- `make check`.
