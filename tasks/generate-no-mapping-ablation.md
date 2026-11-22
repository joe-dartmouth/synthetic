---
id: generate-no-mapping-ablation
title: Add the bko-nomap arm, bko-direct's draw without the mapping table
status: ready
depends_on: --
blocks: evaluate-control-runs-registered, generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-donor-limits, generate-coherent-free-fields, evaluate-covariate-bias-fractions
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/no-mapping-ablation
---

# Add the bko-nomap arm, bko-direct's draw without the mapping table

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`bko-nomap` differs from `bko-direct` only in the mapping table, so the contrast between them
shows what the table carries (ADR
[`research/bkb-direct-generation`](../docs/adr/research/bkb-direct-generation.md) holds the draw
it ablates). The control reading rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md) reads it.

## Scope

**In.**

- A frozen `DrawSpec` dataclass in `bko.generate` with fields `presence` and `mapped` (default
  `True`).
- `DIRECT_ARMS` in `bko.evaluate.experiment`, today a mapping of arm name to `Presence`, becomes
  `Mapping[str, DrawSpec]`.
- Arm `bko-nomap`, at `mapped=False`: no field groups, every column a free field, the same donor
  fallback, seed and trace as `bko-direct`.
- Only non-default spec fields are logged in the `arm` params, so existing arms' lines keep their
  bytes.
- A new ADR under `docs/adr/research/` that records the arm.

**Out.**

- Any run of record.

## Acceptance

- With one fixture cell and a fixed seed, `bko-nomap`'s draw equals a uniform pick per column from
  the cell-else-label-group pool.
- `bko-direct`'s digests are unchanged.
- `trace` reads a `bko-nomap` fill.
- `make check` passes.
