---
id: evaluate-syntheval-results
title: Run the SynthEval readings and regenerate Lautrup et al.'s heart-disease numbers
status: blocked
depends_on: evaluate-syntheval-heart-disease
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the SynthEval readings and regenerate Lautrup et al.'s heart-disease numbers

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Lautrup et al. tabulate SynthEval numbers only for their hepatitis dataset. Their heart-disease
results are figures, so a comparison needs their numbers regenerated from their MIT code
([repository](https://github.com/notna07/disjoint-synthetic-data-generation)). Unit
[`evaluate-syntheval-heart-disease`](evaluate-syntheval-heart-disease.md) lands the scoring
command and the runs.

## Scope

**In.**

- Run the named runs and score each fill with the unit's command.
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
