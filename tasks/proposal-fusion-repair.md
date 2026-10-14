---
id: proposal-fusion-repair
title: Fuse validated records and measure gap closure
status: blocked
depends_on: proposal-llm-generation, proposal-record-validation
blocks: proposal-streamlit-app, proposal-final-evaluation
parallel_safe: true
owner: Shirley
milestone: status-update-2
paid: false
adrs: --
---

# Fuse validated records and measure gap closure

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The completeness criterion asks that at least half the detected gaps close after fusion ([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)). Fusion is in Shirley's workstream while `fuse` lives in `bko.bkb`; this unit settles the interface between the two.

## Scope

**In.**

- Fuse validated records back into the BKO and re-run gap detection to count closed gaps.

**Out.** The downstream prediction comparison.

## Acceptance

- Gap closure is computed from one command and recorded in `docs/EVIDENCE.md`.
- `make check`.
