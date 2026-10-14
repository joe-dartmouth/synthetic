---
id: proposal-mondo-do-fusion
title: Reproduce the paper's MONDO/DO fusion
status: blocked
depends_on: proposal-bkb-port, proposal-ontology-mapping
blocks: proposal-gap-detection
parallel_safe: true
owner: Partiban
milestone: status-update-1
paid: false
adrs: research/bko-formal-gap-definition
---

# Reproduce the paper's MONDO/DO fusion

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The first timeline row of the 2-pager is reproducing Santos et al.'s MONDO/DO fusion ([ROADMAP](../docs/ROADMAP.md#status-update-1)). It proves the BKB layer and the ontology loading work together before the dataset joins.

## Scope

**In.**

- Fuse MONDO and DO as BKO fragments with the BKB layer and compare against what Santos et al. report.

**Out.** The dataset fragment, which joins in `proposal-gap-detection`.

## Acceptance

- The reproduction and any difference from the paper are recorded in `docs/EVIDENCE.md`.
- ADR `research/bko-formal-gap-definition` moves to `accepted` if the reproduction holds.
- `make check`.
