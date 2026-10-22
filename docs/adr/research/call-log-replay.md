---
status: accepted
topic: research
created: 2026-10-19
updated: 2026-10-22
---
# A paid run's call log answers its rerun, and a stopped run's log its resumption, call by call in order

## Context

The Messages API takes no seed (ADR `research/record-generation-design`), so a paid run's records
cannot be drawn again. E1 judges BKO arms whose rows come from such calls, and every number in its
report must be regenerable from a committed command (`docs/EVIDENCE.md`, E1). The call log already
holds every request's parameters and every reply's whole response.

A paid run can stop late. `e1-2` stopped after 83 of 100 holdouts, and finishing it cost 2.15 USD
of new spend against about 12.82 USD for a fresh run (measured 2026-10-21, its and `e1-3`'s Runs
rows in `docs/EVIDENCE.md`).

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
  - an empty log, or one whose first line is not a `run` line of this schema;
  - a line that is not a JSON object;
  - a request, reply or error line with no whole-number `call`, or whose `call` another line of
    its kind repeats;
  - request lines that are not calls 1 to n;
  - a request with no reply or error line, or a reply or error line with no request;
  - a request line with no `params`, a reply line missing its request id, model, stop reason,
    usage, response or attempts, an error line missing its error or attempts, or a usage whose
    keys are not `Usage`'s or whose counts are not whole numbers;
  - for a resume only: an `error` line before the stopped log's last call; a stopped log whose
    `run` line names `replay`; a resumed log that holds fewer answered calls than it resumed.

  Every check but the first two runs when the log is read, before any call.
- `bko.generate.replay.Resume` is a `Replay` over a stopped log's answered calls; after them, it
  sends live. It reads the stopped log in stopped mode: it drops a final segment with no newline,
  and drops a last call whose answer is not exactly one `reply` line. The resumed run's log, NEW,
  names `resume: {sha256, calls}` in its `run` and `end` lines, and its `end` line carries
  `live_usd`. `score --replay NEW` writes the resumed run's scores byte for byte.
- `python -m bko.evaluate score --replay LOG --log NEW` answers every BKO call from `LOG` and
  writes its own log to `NEW`. That log's `run` line names `LOG`'s sha256 under `replay`, so a
  replay's log never reads as paid spend; its `live_usd` is 0. The run refuses a log with calls
  left unplayed. `score --resume STOPPED --log NEW` answers STOPPED's answered calls and sends the
  rest live; NEW's `usd` counts its replayed calls, and its `live_usd` counts what it paid.
- `bko.app` answers a BKO fill from the call log of an earlier fill by that arm, with the same
  refusals, and writes its own log naming the replayed log's sha256.
- Order, not content, pairs a call with its answer.

## Rationale

A BKO arm sends the same gap request on every holdout and on each retry. At level `ontology`,
`bko`'s two gap-request prompts are identical across every E1 holdout (measured 2026-10-19). One
request is therefore logged many times with different replies, and only the call order tells them
apart.

Everything after a call is deterministic: decoding, validation, drops, top-ups, numbering and the
classifier. Answering the same calls in the same order therefore rewrites the scores byte for byte.

Comparing each request with its logged one turns a code change that moves a prompt, a quota or a
top-up into a refusal, instead of a silently different reading.

A resumed run's log holds every call, replayed and live, in one file, so its scores replay from
one log as any paid run's do.

## Consequences

**Gained**: E1's BKO scores regenerate byte for byte from the paid run's log at its commit, with no
API call, and a fix to a free reading re-records for nothing.
**Accepted**: a change to a prompt, a quota, a drop rule that moves a top-up, or the settings makes
a log unreplayable and needs a new paid run; replay holds on one machine at the run's commit, data
and releases, as E1's baselines do. Decontamination makes every arm's scores depend on every arm's
rows, so byte-identical replay also needs the seeded baselines to repeat, as they do on one
machine. A resume pays its stopped call again. The spend ceiling counts replayed spend, so a run
stopped at the ceiling cannot resume. A stop that recurs after the last call recurs on resume, at
CPU cost only. Nothing locks a log another process still writes. A missing credential and drifted
settings show at the first live or replayed call.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Replaying from the candidates file | It carries no request to check and loses the top-up rounds | 2026-10-19 |
| Looking a reply up by its request | The same request is logged with many replies | 2026-10-19 |
| Recording each fill's rows instead of its calls | Regenerates utility but not validity's counts or the drops | 2026-10-19 |
| Rerunning a stopped run from scratch | Repays every answered call; finishing e1-2 that way would have cost about 12.82 USD against 2.15 USD | 2026-10-22 |
| Appending to the stopped log | A paid log is created exclusively and never extended, and the stopped log stays the record of the stopped run | 2026-10-22 |
| A resumed log holding only its live calls | Its scores would replay from no single file | 2026-10-22 |

## Enforcement

- Owner: `src/bko/generate/replay.py`
- Pinned by: `tests/test_generate_replay.py`, `tests/test_evaluate_bko_cli.py`,
  `tests/test_app_steps.py`

## Reversal Conditions

**The Messages API takes a seed**: a paid run then reproduces by drawing again.

**A replay's scores differ from its paid run's at the same commit, or a resumed run's replay
differs from the resumed run**: something after the call is
unseeded, and that code changes, not this record.

## Related

- [Record generation](record-generation-design.md)
- [E1's evaluation](e1-evaluation-protocol.md)
