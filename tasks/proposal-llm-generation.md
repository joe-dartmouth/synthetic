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
- Fill each cell of a gap request, which in E1 is count-based and the same for every arm (ADR
  `research/e1-evaluation-protocol`). Guide each cell's records by the ontology relationships of
  the (cell, statement) gaps the caller passes for it, under any definition of ADR
  `research/bko-formal-gap-definition`; a cell the caller passes no gap for is guided by the
  relationships of the concepts its training rows show.
- Hand validation every candidate a call yields, unfiltered and unrepaired, with one placeholder for
  each record the call asked for and did not decode, as ADR `research/record-validation-design`
  (Decision, Input) states, and pin that with a test.

**Out.** Validation and fusion.

## Acceptance

- Port mht3's paid-API request guard (`refuse_unmarked_api_requests` in its `tests/conftest.py`), so
  an unmarked test cannot reach the API once the `llm` extra joins `make setup`.
- Tests that call the API are marked `live` and run only with `--live`.
- Each run's API spend and request/response log location are recorded in the Runs field of its
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry.
- `make check`.
