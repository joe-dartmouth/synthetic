---
id: proposal-llm-generation
title: Record generation through the Messages API
status: ready
depends_on: --
blocks: proposal-fusion-repair
parallel_safe: true
owner: Shirley
milestone: status-update-2
paid: true
adrs: --
---

# Record generation through the Messages API

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Generate: candidate records for each gap through the Anthropic Messages API, guided by the relevant ontology relationships ([RESEARCH, Generate and validate](../docs/RESEARCH.md#generate-and-validate)).

## Scope

**In.**

- Prompted generation of candidate records per gap in `src/bko/generate/`, with the request and
  response logged per run.

**Out.** Validation and fusion.

## Acceptance

- Port mht3's paid-API request guard (`refuse_unmarked_api_requests` in its `tests/conftest.py`), so
  an unmarked test cannot reach the API once the `llm` extra joins `make setup`.
- Tests that call the API are marked `live` and run only with `--live`.
- Each run's API spend and request/response log location are recorded in the Runs field of its
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry.
- `make check`.
