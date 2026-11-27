---
id: ontology-auto-mapping-results
title: Run the automatic mappings against the hand mapping on the development frames
status: ready
depends_on: --
blocks: paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/automatic-mapping
---

# Run the automatic mappings against the hand mapping on the development frames

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

An automatic BKO matters only if its rows are as useful as the hand-built one's.

ADR [`research/automatic-mapping`](../docs/adr/research/automatic-mapping.md) records the
builders, the measures and the permuted control, and the
[E22 entry](../docs/EVIDENCE.md#e22-automatic-mappings-on-the-diabetes-130-removal) of
`docs/EVIDENCE.md` holds the Configuration: the runs, their families, datasets and arms. `RUNS`
holds them (`bko.evaluate.runs_direct.E22` and `E22_FAMILIES`).

Each table's precision, recall, exact matches, hierarchy hits, unmovable rows and judged counts
against the hand rows come from

    uv run --no-sync python -m bko.ontology.auto_measures </dev/null

which reads the committed tables of `bko.ontology.auto_tables` and the pinned releases, and makes
no API call.

## Scope

**In.**

- The sixteen E22 runs and their slices.
- Each mapping's measures against the hand tables, from the command above: exact matches
  reported apart from `hierarchy_hits`, a hit on an ancestor or a descendant of the hand concept,
  which `precision_recall` counts as a miss.
- Automatic minus `bko-nomap`, and automatic minus hand under the registered margin.
- The permuted-hierarchy control.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Claims beyond the development frames.

## Acceptance

- The entry is `complete`.
- `make check` passes.
