---
id: gaps-is-a-conditional-fragments
title: Encode each mapped is_a pair's conditional share in the dataset fragment
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: generate-is-a-conditional-draws
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Encode each mapped is_a pair's conditional share in the dataset fragment

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A dataset fragment holds one prior per concept (ADR
[`research/dataset-fragment-encoding`](../docs/adr/research/dataset-fragment-encoding.md)), so the
fused BKB cannot say how often a broad concept accompanies its narrow one. Unit
[`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md) draws from that
conditional.

## Scope

**In.**

- `bko.gaps.observe` adds one conditional S-node per mapped `is_a` pair, its head the broad
  concept and its tail the narrow one, at the records' conditional share.
- `bko.bkb.fusion.check_fused` accepts the conditional S-node.
- `bko.gaps.shares.nested` generalised to a topological order over the mapped `is_a` pairs.
- ADRs
  [`research/dataset-fragment-encoding`](../docs/adr/research/dataset-fragment-encoding.md) and
  [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md)
  revised in place. The first's in-place revision narrows its Ruled Out row on conditional
  S-nodes between concepts that occur together to pairs with no `is_a` relation.

**Out.**

- Any sampler change.
- The gap definitions, which read no conditional.

## Acceptance

- A fixture fragment's conditional equals the hand count.
- Every gap listing of `python -m bko.gaps uci` is byte-equal before and after.
- `bko-direct`'s digests (`DRAW_DIGEST`, `ADDED_DIGEST`) are unchanged.
- `rows` on one existing E6 slice still reproduces it (`matches_folder=true`).
- `make check` passes.
