---
id: proposal-streamlit-app
title: Streamlit interface
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Krish
milestone: working-prototype
paid: true
adrs: --
---

# Streamlit interface

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The Working Prototype milestone is a Streamlit UI over the pipeline
([ROADMAP](../docs/ROADMAP.md#working-prototype), [ARCHITECTURE](../docs/ARCHITECTURE.md)).

## Scope

**In.**

- A Streamlit app in `src/bko/app/`: choose a dataset, analyze gaps, fill them, evaluate, and
  download the result.
- A Make target that launches it.

**Out.** New pipeline logic; the app calls the stages.

## Acceptance

- The app runs locally from one command against UCI Heart Disease.
- Tests that drive generation through the app are marked `live`; the rest use recorded responses.
- The `app` extra joins `make setup` only if tests import it.
- `make check`.
