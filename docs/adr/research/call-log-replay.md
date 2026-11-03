---
status: accepted
topic: research
created: 2026-10-19
updated: 2026-11-02
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
  them. Calls are numbered in dispatch order, which is plan order, so a concurrent run (ADR
  `research/concurrent-call-waves`) numbers its calls as a sequential one does. Log lines are
  written as calls land, and batches are committed in call order. Call i must send the
  parameters of the log's i-th `request` line, compared as JSON. Its answer is one of two things:
  - that call's `reply` line, rebuilt as a `Reply`, its text read from the logged response by
    `bko.generate.messages.text_of`, the function `bko.generate.api.reply_of` and
    `bko.generate.local.reply_of` also use. It reads by shape: a Messages body's text blocks
    joined, thinking blocks ignored, or a chat-completions body's first choice's
    `message.content`, reasoning ignored; None when there is none;
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
    `run` line names `replay`; a stopped log whose `run` line reads `rehearsal: true`, since its
    replies are canned and a resume sends the rest live; a stopped log whose `run` line names
    another run under `run`; a stopped log whose `end` line reads `stop` `projection` or
    `ceiling` and has no raised ceiling (below), names no run, records no finished `holdouts`, or
    records a `usd` or `live_usd` that is not a finite number of at least 0, or whose raised
    ceiling is not above its projection; a raised ceiling for any other stop; a resumed log that
    holds fewer answered calls than it resumed. A log with no `stop` field predates it and
    resumes, as does one with no `run` field that no spend rule stopped.
    `bko.generate.resume_guard.refuse_unresumable` holds the checks on the `run` and `end` lines,
    and `launch --resume` runs them on those two lines alone before it claims a folder.

  Every check but the first two runs when the log is read, before any call.
- A ceiling or projection stop resumes only under a ceiling raised for that resume alone,
  `--ceiling USD`, above the stopped run's own projection: the `end` line's `usd` over its
  `holdouts` (at least 1), scaled to the named run's holdouts, and never below `usd`, as
  `bko.evaluate.watch` scales a run's spend. The `end` line of such a stop records `holdouts`, the
  holdouts finished. A refusal names the logged spend, its live part, the projection and the
  ceiling. The resumed run's log adds `stop` and `ceiling` to its `resume` field, `score` logs a
  `resume stop=` line and `launch.log` a `ceiling=` line. The named run, its `experiment` line and
  the code stay as launched, so `merge` still joins a resumed slice to its siblings. Any later
  resume of that log's run runs under the ceiling its log ran under (`resume_guard.ran_under`),
  recorded as `ceiling` alone, so a supervisor's resume of an API failure keeps the raise. The
  supervisor never resumes a ceiling or projection stop; an operator's `launch --resume` does.
  `launch --resume` refuses a stopped log whose sha256 another run folder's `run` line already
  names under `resume`, since resuming an earlier segment would send calls a later one paid for.
- `bko.generate.replay.Resume` answers a stopped log's answered calls as `Replay` does; after
  them, it sends live. It reads the stopped log in stopped mode: it drops a final segment with no
  newline, and drops a last call whose answer is not exactly one `reply` line. For a log whose
  `run` line holds `in_flight` above 1, it drops every call from the first one not answered by
  exactly one `reply` line, with one warning each. The new run's spend (`carried_usd` in its
  `run` line) carries the dropped calls' `usd` plus what the stopped log itself carried, at any
  in-flight, so the ceiling reads the whole run. A log without `in_flight` keeps the last-call
  rule, and an `error` line before its last call is refused. The resumed run's log, NEW, names
  `resume: {sha256, calls}` in its `run` and `end` lines, and its `end` line carries `live_usd`.
  `score --replay NEW` writes the resumed run's scores byte for byte.
- `python -m bko.evaluate score --replay LOG --log NEW` answers every BKO call from `LOG` and
  writes its own log to `NEW`. That log's `run` line names `LOG`'s sha256 under `replay`, so a
  replay's log never reads as paid spend; its `live_usd` is 0. The run refuses a log with calls
  left unplayed. `score --resume STOPPED --log NEW` answers STOPPED's answered calls and sends the
  rest live; NEW's `usd` counts its replayed calls, and its `live_usd` counts what it paid.
  `launch <run-id> [<run>] --resume STOPPED` starts that command under the paid-run protocol
  (ADR `research/e1-evaluation-protocol`). A rehearsal's log (`score --rehearse`) replays like any
  other, and the replay's own `run` line keeps `"rehearsal": true`, so it never reads as a paid
  run's replay.
- `bko.app` answers a BKO fill from the call log of an earlier fill by that arm, with the same
  refusals, and writes its own log naming the replayed log's sha256.
- Order, not content, pairs a call with its answer.
- A replay runs at the run's commit (its `versions` line names it), or at a replay commit: the
  run's commit plus changes to the reply-text reader alone, so
  `git diff --stat <run commit> <replay commit> -- src` lists only `src/bko/generate/messages.py`
  and `src/bko/generate/local.py`. A replay commit carries a tag `replay/<name>` pushed to
  origin; G6's local runs replay at `replay/g6-local`.

## Rationale

A BKO arm sends the same gap request on every holdout and on each retry. At level `ontology`,
`bko`'s two gap-request prompts are identical across every E1 holdout (measured 2026-10-19). One
request is therefore logged many times with different replies, and only the call order tells them
apart. At `summary`, `SETTINGS`' level, each holdout's prompt carries its own spreads and bounds,
so prompts differ across holdouts, and pairing by call order is unaffected.

Everything after a call is deterministic: decoding, validation, drops, top-ups, numbering and the
classifier. Answering the same calls in the same order therefore rewrites the scores byte for byte.

Comparing each request with its logged one turns a code change that moves a prompt, a quota or a
top-up into a refusal, instead of a silently different reading.

A resumed run's log holds every call, replayed and live, in one file, so its scores replay from
one log as any paid run's do.

One reader for both reply shapes gives a local run's replay the text its live run decoded, since a
chat-completions body has no `content` list. The reader runs before decoding and changes no
request, so a replay commit that differs from the run's commit in the reader alone answers the same
calls and rewrites the same bytes; the `diff --stat` guard shows that nothing else moved. The
replay commit sits off `main`, and the tag keeps it reachable.

## Consequences

**Gained**: E1's BKO scores regenerate byte for byte from the paid run's log at its commit, with no
API call, and a local run's from its log at its replay commit, and a fix to a free reading re-records for nothing.
**Accepted**: a change to a prompt, a quota, a drop rule that moves a top-up, or the settings makes
a log unreplayable and needs a new paid run; replay holds on one machine at the run's commit or replay commit, data
and releases, as E1's baselines do. Decontamination makes every arm's scores depend on every arm's
rows, so byte-identical replay also needs the seeded baselines to repeat, as they do on one
machine. A resume pays its stopped calls again: one for a sequential run, a stopped wave's for a
concurrent one. The spend ceiling counts replayed spend, so a run stopped at the ceiling resumes
only under a raised ceiling, and the resumed run's projection is read again over the whole run. A
stop that recurs after the last call recurs on resume, at
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
| Rerunning the paid runs to reproduce them at a later commit | Repeats runs whose results exist and draws new rows; a replay commit replays their logs for free | 2026-11-01 |
| Never resuming a ceiling or projection stop, a rerun taking a new run id | User decision for E6's capped slices: a resume under a raised ceiling recovers the calls already paid for, and the projection check keeps the raise above the run's own projection | 2026-11-02 |
| Raising the ceiling in the named run, by a commit | The commit moves the resumed slice's `versions` line, so `merge` refuses it, and an E6 fraction's ceiling sets every slice's; `--ceiling` raises one resume's | 2026-11-02 |

## Enforcement

- Owner: `src/bko/generate/replay.py`, `src/bko/generate/resume_guard.py`
- Pinned by: `tests/test_generate_replay.py`, `tests/test_generate_resume_guard.py`,
  `tests/test_evaluate_bko_cli.py`, `tests/test_evaluate_raised_ceiling.py`,
  `tests/test_evaluate_launch.py`, `tests/test_evaluate_supervise.py`,
  `tests/test_app_steps.py`, `tests/test_generate_local.py`

## Reversal Conditions

**The Messages API takes a seed**: a paid run then reproduces by drawing again.

**A replay's scores differ from its paid run's at the same commit, or a resumed run's replay
differs from the resumed run**: something after the call is
unseeded, and that code changes, not this record.

## Related

- [Record generation](record-generation-design.md)
- [E1's evaluation](e1-evaluation-protocol.md)
