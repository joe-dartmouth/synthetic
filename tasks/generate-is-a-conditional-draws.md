---
id: generate-is-a-conditional-draws
title: Add the bko-hier arms, drawing a broad concept given its narrow one
status: blocked
depends_on: generate-no-mapping-ablation, gaps-is-a-conditional-fragments, evaluate-added-row-contradictions
blocks: evaluate-variant-dev-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add the bko-hier arms, drawing a broad concept given its narrow one

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`bko-direct`'s narrow-without-broad draws follow the data's contradictions, and no arm lets the
ontology weigh against them short of forcing the closure. ADR
[`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md) rules
out forcing ancestors into fields as the default. This unit adds arms that shrink the data's
conditional toward the `is_a` S-node, with the forced closure, `bko-hier-closed`, as an ablation
arm beside them.

## Scope

**In.**

- `DrawSpec.hierarchy: float | None`, a pseudo-count.
- Arms `bko-hier-0` (pseudo-count 0), `bko-hier-w` (`GROUP_WEIGHT`, `bko.gaps.shares`) and
  `bko-hier-closed` (the closure forced).
- A broad concept is drawn given its narrow one at the fragment's conditional (unit
  [`gaps-is-a-conditional-fragments`](gaps-is-a-conditional-fragments.md)) shrunk toward the `is_a`
  S-node's probability 1.
- ADR [`research/bkb-direct-generation`](../docs/adr/research/bkb-direct-generation.md) revised in
  place.
- ADR [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md)
  revised in place: its Ruled Out row on forcing ancestors into fields is scoped to `bko-direct`,
  and the ADR records `bko-hier-closed` as an ablation arm only. Decision 7 still holds for
  `bko-direct`.
- Development `RUNS` entries on E6's and E3's frames, and the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`. Its
  target measure is the narrow-without-broad count of unit
  [`evaluate-added-row-contradictions`](evaluate-added-row-contradictions.md).
- The Configuration states that each frame holds one or two such pairs, so the arm is a
  contradiction-handling ablation.

**Out.**

- Rejecting draws.
- Changing `bko-direct`.

## Acceptance

- Pseudo-count 0 reproduces the pair's training joint on a fixture.
- The closed arm yields no narrow-without-broad row.
- `bko-direct`'s digests are unchanged.
- `make check` passes.
