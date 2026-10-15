---
id: proposal-bkb-port
title: Bayesian Knowledge Base layer from pybkb
status: ready
depends_on: --
blocks: proposal-mondo-do-fusion
parallel_safe: true
owner: Partiban
milestone: status-update-1
paid: false
adrs: research/bkb-layer-source
---

# Bayesian Knowledge Base layer from pybkb

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

2-pager section 4 names the absence of a public BKB library as the principal risk. pybkb exists
([ARCHITECTURE](../docs/ARCHITECTURE.md#pybkb)) but pins Python ~3.11, so it cannot be a dependency
here; its core can be ported. Everything that detects a gap needs this layer
([RESEARCH section 3](../docs/RESEARCH.md#3-approach)).

## Scope

**In.**

- First, the source decision: port pybkb's code under the repository's GPL-3.0 `LICENSE` (ADR
  `tooling/gpl-3-license`), or implement clean-room from the papers with pybkb as a test oracle in a
  separate Python 3.11 environment. A new ADR under `docs/adr/research/` records it.
- The BKB structure, `fuse` (source-weighted fusion) and the `BKBReasoner.update` path, plus the
  probability utilities they need, in `src/bko/bkb/` on Python 3.14 with numpy 2.

**Out.** pybkb's `learn/` (gobnilp and gurobi) and its cytoscape visualization mixin.

## Acceptance

- On the port path, pybkb's example `.bkb` files load, fuse and answer an update query with the same
  result as pybkb. On the clean-room path, pybkb runs only as an oracle outside the repository, and
  neither its code nor its example files are committed.
- `make check`.
