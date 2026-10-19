---
status: accepted
topic: research
created: 2026-10-19
updated: 2026-10-19
---
# A paid run's call log answers its rerun, call by call in order

## Context

The Messages API takes no seed (ADR `research/record-generation-design`), so a paid run's records
cannot be drawn again. E1 judges BKO arms whose rows come from such calls, and every number in its
report must be regenerable from a committed command (`docs/EVIDENCE.md`, E1). The call log already
holds every request's parameters and every reply's whole response.

## Decision Drivers

* **Must regenerate a paid run's statistics without spending**, so a fix to a free reading costs
  nothing.
* **Must refuse a rerun that asks something the log did not**, so drift shows rather than reads as
  a result.
* **Must keep `anthropic` inside `bko.generate.api`** (ADR `research/record-generation-design`).

## Decision

- `bko.generate.replay.Replay` is a `Send` over a call log's calls, in the order the log made
  them. Call i must send the parameters of the log's i-th `request` line, compared as JSON. Its
  answer is one of two things:
  - that call's `reply` line, rebuilt as a `Reply`, its text read from the logged response by
    `bko.generate.messages.text_of`, the function `reply_of` also uses;
  - or that call's `error` line, raised again as `GenerationError`.

  `ReplayError` (exit 3) is raised for:
  - a call past the log's end;
  - a call that differs from the logged request;
  - a line that is not a JSON object;
  - a log of another schema;
  - a request with no reply or error line.
- `python -m bko.evaluate score --replay LOG --log NEW` answers every BKO call from `LOG` and
  writes its own log to `NEW`. That log's `run` line names `LOG`'s sha256 under `replay`, so a
  replay's log never reads as paid spend. The run refuses a log with calls left unplayed.
- Order, not content, pairs a call with its answer.

## Rationale

A BKO arm sends the same request on every holdout and on each retry. At level `ontology`, `bko`'s
two prompts are identical across E1's 100 holdouts (measured 2026-10-19). One request is therefore
logged many times with different replies, and only the call order tells them apart.

Everything after a call is deterministic: decoding, validation, drops, top-ups, numbering and the
classifier. Answering the same calls in the same order therefore rewrites the scores byte for byte.

Comparing each request with its logged one turns a code change that moves a prompt, a quota or a
top-up into a refusal, instead of a silently different reading.

## Consequences

**Gained**: E1's BKO scores regenerate byte for byte from the paid run's log at its commit, with no
API call, and a fix to a free reading re-records for nothing.
**Accepted**: a change to a prompt, a quota, a drop rule that moves a top-up, or the settings makes
a log unreplayable and needs a new paid run; replay holds on one machine at the run's commit, data
and releases, as E1's baselines do.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Replaying from the candidates file | It carries no request to check and loses the top-up rounds | 2026-10-19 |
| Looking a reply up by its request | The same request is logged with many replies | 2026-10-19 |
| Recording each fill's rows instead of its calls | Regenerates utility but not validity's counts or the drops | 2026-10-19 |
| Resuming a failed run by replaying its log, then calling live | Needs a judged-run rule for a run made in two sessions | 2026-10-19 |

## Enforcement

- Owner: `src/bko/generate/replay.py`
- Pinned by: `tests/test_generate_replay.py`, `tests/test_evaluate_bko_cli.py`

## Reversal Conditions

**The Messages API takes a seed**: a paid run then reproduces by drawing again.

**A replay's scores differ from its paid run's at the same commit**: something after the call is
unseeded, and that code changes, not this record.

## Related

- [Record generation](record-generation-design.md)
- [E1's evaluation](e1-evaluation-protocol.md)
