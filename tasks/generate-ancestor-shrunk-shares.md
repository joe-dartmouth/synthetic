---
id: generate-ancestor-shrunk-shares
title: Add the bko-tree arm, drawing a coded field's shares shrunk along the ontology
status: blocked
depends_on: generate-no-mapping-ablation, ontology-diabetes-diagnosis-codes
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/ancestor-shrunk-shares
---

# Add the bko-tree arm, drawing a coded field's shares shrunk along the ontology

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

With a fine-grained coded field, most codes are rare in a small cell. The ontology says which
codes are alike, and `bko-direct`'s sampler ignores it. This arm lets a code borrow strength from
its ontology neighbours.

## Scope

**In.**

- `DrawSpec.tree: bool` and arm `bko-tree`.
- A coded field whose rows partition its codes is drawn as one categorical per record.
- A code's share in a cell is its cell share shrunk, with weight `GROUP_WEIGHT`
  (`bko.gaps.shares`), toward its parent class's share in the cell times the code's share within
  that parent in the label group. The rule applies recursively from the release's root down. A
  parent's share is the share of records showing any of its descendants.
- The trace names each level's parts.
- A new ADR under `docs/adr/research/` that records the rule.
- Development `RUNS` entries on `diabetes-codes` (unit
  [`ontology-diabetes-diagnosis-codes`](ontology-diabetes-diagnosis-codes.md)) with E6's removal,
  for `bko-tree`, `bko-direct` and `bko-nomap`, and the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`. Its
  target measure is removed-cell recall, with the per-code share error against the held-out cell
  as a second reading.

**Out.**

- Fields whose rows overlap.
- Changing `bko-direct`.

## Acceptance

- On a fixture tree, a code unseen in the cell gets a share above 0 and below its parent's.
- Shares over a field's codes sum to 1.
- `bko-direct`'s digests are unchanged.
- `make check` passes.
