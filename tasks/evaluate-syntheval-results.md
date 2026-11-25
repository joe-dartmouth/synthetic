---
id: evaluate-syntheval-results
title: Run the SynthEval readings and regenerate Lautrup et al.'s heart-disease numbers
status: ready
depends_on: --
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/syntheval-reading
---

# Run the SynthEval readings and regenerate Lautrup et al.'s heart-disease numbers

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Lautrup et al. tabulate SynthEval numbers only for their hepatitis dataset. Their heart-disease
results are figures, so a comparison needs their numbers regenerated from their MIT code
([repository](https://github.com/notna07/disjoint-synthetic-data-generation)). ADR
[`research/syntheval-reading`](../docs/adr/research/syntheval-reading.md) records the scoring
command, the `hd` mapping and the run `p2`, and the
[P2 entry](../docs/EVIDENCE.md#p2-uci-heart-disease-scored-by-syntheval) of `docs/EVIDENCE.md` its
Configuration.

## Scope

**In.**

- Run `p2`, then `rows` over its folder, and score each replica fill with `syntheval`.
- Regenerate Lautrup et al.'s numbers from their code and codebooks on their `hd` split. Their
  code runs in a scratch environment outside the lock, its commit recorded.
- The EVIDENCE entry's Runs, Result and Read against, recording both.

**Out.**

- Any change to the command, the mapping or an arm.

## Acceptance

- The entry's status is `complete`.
- The entry's Read against compares with the heart-disease figures of Lautrup et al., beside the
  numbers regenerated from their code.
- Each run's sha256 is recorded.
- `make check` passes.
