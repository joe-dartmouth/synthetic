---
id: proposal-run-browser
title: Earlier runs in the Streamlit interface
status: blocked
depends_on: proposal-streamlit-app
blocks: --
parallel_safe: true
owner: Krish
milestone: working-prototype
paid: false
adrs: --
---

# Earlier runs in the Streamlit interface

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The Working Prototype demo fills a small use case live; the larger runs of record are shown from
their run folders ([`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#runs)) in the same interface.

## Scope

**In.**

- A view in `bko.app` that lists the run folders under `runs/` and shows each one's stored
  artifacts by role: E1's criteria, means and validity; C1's candidates and closure; G1's
  candidates; E2's gap tables; R1's reproduction; each run's call-log totals; the demo's fills.

**Out.** Recomputing anything from `data/`; runs without a run folder.

## Acceptance

- The view lists every run folder present and shows each role it holds.
- A missing `runs/`, a missing role or an unreadable artifact shows a message, not a traceback.
- `make check`.
