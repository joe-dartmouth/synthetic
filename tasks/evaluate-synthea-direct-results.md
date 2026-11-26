---
id: evaluate-synthea-direct-results
title: Run bko-direct and the controls on Synthea beside E3
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run bko-direct and the controls on Synthea beside E3

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

No Synthea run includes `bko-direct`
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e3-synthea-downstream-augmentation)), so the
development result on that frame is missing.

## Scope

**In.**

- The Synthea run [E13](../docs/EVIDENCE.md#e13-control-arms-on-the-synthea-removal) names, and its readings, as
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives them.
- `compare --beside` against `runs/e3-7/`'s `bko` and the CLLM arms.
- The EVIDENCE entry's Result and Read against, naming the branch of the control reading rule in
  [`docs/RESEARCH.md`](../docs/RESEARCH.md) that each contrast falls in.

**Out.**

- Paid calls.

**Findings from planning.**

- `runs/e3-7` holds `scores.csv`, `criteria.csv` and `criteria.log`, and no `scores.log` or `rows`
  store, so this unit's first step confirms that `compare --beside runs/e3-7/scores.csv` reads it.

## Acceptance

- The entry is `complete`, with each contrast's branch named.
- `make check` passes.
