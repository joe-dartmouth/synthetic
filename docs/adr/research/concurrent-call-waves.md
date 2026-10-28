---
status: accepted
topic: research
created: 2026-10-27
updated: 2026-10-27
---
# A round's independent calls go out as concurrent waves that keep the call plan

## Context

A paid E1-shaped run is serial: each call waits for the one before it. On `runs/g3-5`, API wait is
most of the wall time: the gaps between each call's `request` and `reply` times in its
`calls.jsonl`, summed, come close to the span from its `run` line's `started` to its `end` line's
`ended`, and local work is small beside them. Nothing runs concurrently inside a process.

A call's request depends on the records kept in its cell, and a call log replays call by call in
order (ADR `research/call-log-replay`). Concurrency must therefore leave the call plan, its
numbering and its stop calls as a sequential run has them, or old logs stop replaying and a
ceiling stop lands elsewhere.

## Decision Drivers

* **Must keep the call plan**, so a log written at any in-flight replays to the same bytes.
* **Must stop where a sequential run stops**, at the spend ceiling and at the idle limit.
* **Must keep `anthropic` inside `bko.generate.api`** (ADR `research/record-generation-design`).
* **Should cut wall time**, since the API wait dominates it.

## Decision

- **Wave rule.** Each round keeps its budget check. It walks the short cells in quota order. A
  wave is the longest run of consecutive cells with distinct (sex, label) chains, and a repeated
  chain closes the wave. A cell's rows are read from committed state with `attempt.short()` and
  chunked as before. The wave is sent, its batches are committed in call order, then the next wave
  is built.
- **Why the plan holds.** A cell's ask depends only on records kept in it. Structured outputs pin
  sex and label, so only cells of one chain interact, and they stay ordered across waves.
  Requests are dispatched in plan order, so call numbers match a sequential run's, and
  `Replay.__call__` answers in dispatch order.
- **Guard.** If a take keeps a row in a cell that a later group of the same wave already asked
  for, the run raises `WaveError(ContractError)` and exits 3.
- **Client.** `AsyncAnthropic(middleware=[Attempts()], max_retries=MAX_RETRIES, timeout=TIMEOUT)`.
  `Attempts` implements `handle_async`, which the SDK requires of an async client's middleware, and
  keeps per-call counts in a `ContextVar` holding a mutable holder.
- **In flight.** `score --in-flight N` sets the calls in flight; the default is
  `bko.generate.flight.IN_FLIGHT`. It is recorded in the call log's `run` line as `in_flight`
  (additive; the schema stays 2) and on the `log path=` line. It is not in `Settings`: the
  candidates header and the resume check read `Settings`, and in-flight changes nothing a call
  sends. `launch` forwards it only when given.
- **Ceiling.** Before call j, reserve each lower-numbered in-flight call's
  `max_tokens x PRICES.output + (len(json.dumps(params)) + 2000) x max(PRICES.input,
  PRICES.cache_write)`, so a call billed as a cache write is not understated. If the log's
  spend plus the reservations reaches the ceiling, wait for those calls, then check exactly. The
  stop lands at the call an in-flight 1 run stops at whenever each reservation bounds its call's
  bill. A reservation bounds one attempt: an SDK retry after a timeout the server billed can bill
  past it. The log sums spend in landing order, so its total can differ from a call-order sum in
  the last float digits. A ceiling stop outranks a call in flight that then fails, so the call
  log's `end` line still names `ceiling` and the run is not resumed.
- **Idle.** `asyncio.timeout(idle_limit)` wraps each awaited send, retries included. On expiry the
  call's error line carries its attempt count, and the run sends no later call. The calls already
  in flight land under their own idle limits, at most one more limit of wall time, so the replies
  and spend the server may bill are recorded. The run then raises `StopRuleError("idle")` for the
  lowest idle call. The stop is raised from inside the event loop, with no signal.
- **Interrupt.** A `KeyboardInterrupt` cancels the calls in flight. Every requested call gets an
  error line before the call log closes, one cancelled before it started and one hit by a further
  Ctrl-C included.
- **Resume.** For a log whose `run` line holds `in_flight` above 1, a resume drops every call from
  the first one not answered by exactly one `reply` line, with one warning each, and carries the
  dropped calls' `usd` into the new run's spend as `carried_usd`. At any in-flight,
  `carried_usd` also holds what the stopped log itself carried. Logs without the key keep the
  last-call rule (ADR `research/call-log-replay`).
- **Log identity.** The `run status=` line names `run=`, `level=` and `in_flight=`; `compare`'s
  `beside` lines name `run=` and `level=`; each 429 or retryable status with an attempt left logs
  `retry call=%d attempt=%d status=%s retry_after=%s`. This answers a rate-limit storm that trips
  the idle rule with a line per retry.
- **Supervised launch.** `launch` supervises its run and resumes API-caused stops. It starts
  `python -m bko.evaluate supervise`, which runs `score` as a child process, one segment at a time.
  The call log's `end` line says under `retryable` whether the API caused the stop: true for an idle
  stop and for a failure from a connection error, a timeout, a 408, 409 or 429 (the statuses below
  500 the SDK retries), or a status of 500 or above; false otherwise, ceiling and projection stops
  included. It says under `rate_limited` whether any retry in the segment saw a 429 (both keys
  additive; the schema stays 2). A `retryable` stop, or a child a signal killed with no `end` line,
  is resumed into `<id>-resume-<k>` after `min(60 x 2^(k-1), 900)` seconds, at most `--retries`
  times, at half the in-flight (at least 1) when the segment was rate limited; a child that exits on
  its own with no `end` line crashed and is not resumed. Each resume continues the last segment's
  call log, which holds every call, so the last segment's log replays the whole run. An operator's
  signal to the supervisor, forwarded once to the child as `SIGINT`, or a `STOP` file in the run's
  folder stops the run with no resume (`bko.evaluate.stopping`).
- **Callers.** `guided_arms` takes the run's name and its limits. The generate CLI records
  `run=generate` at in-flight 1 and stays sequential. The app sends through the async client at
  in-flight 1 with the experiment's idle limit and `run=app`, since a synchronous send cannot be
  bounded by `asyncio.timeout`.

## Rationale

Waves are the largest unit that leaves the plan unchanged. Distinct chains cannot see each other's
kept rows, so their calls commute; a repeated chain cannot, so it waits for the batch before it.
Reserving before dispatch, then waiting and checking exactly, makes the ceiling a function of the
plan and not of reply timing. An `asyncio.timeout` per awaited send bounds a call from inside the
loop, where a signal cannot reach a worker thread or Streamlit's script thread.

The gain is a simulated estimate over `g3-5`'s recorded latencies: several-fold at 8 in flight,
for E1's shape and E3's alike.

## Consequences

**Gained**: wall time falls by the simulated gain above; golden logs replay at any in-flight; the
idle stop needs no signal; retries are attributed to their call; concurrent logs name their run and
level.
**Accepted**: a rate-limited organisation may see 429s at high in-flight, which the SDK's retries
absorb and the idle rule turns into a stop if they persist; a resume of a concurrent run pays its
stopped wave's calls again.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Threads | A stop would hang at interpreter exit | 2026-10-27 |
| Whole-round waves | They change some of `g3-5`'s rounds' asks, so the plan moves | 2026-10-27 |
| Arms concurrently | An arm's first call number depends on the arms before it | 2026-10-27 |
| Message Batches | They cost many sequential waves, or a lockstep rewrite | 2026-10-27 |
| Parallel local fits | They seed process-wide generators and save little time | 2026-10-27 |
| A resume loop inside `score` | One process would hold the call log across segments, so a killed or slept process loses every segment | 2026-10-27 |

## Enforcement

- Owner: `src/bko/generate/flight.py`; supervision: `src/bko/evaluate/supervise.py` and
  `src/bko/evaluate/stopping.py`
- Pinned by: `tests/test_generate_flight.py`, `tests/test_evaluate_waves.py`,
  `tests/test_evaluate_supervise.py`, `tests/test_evaluate_stopping.py`

## Reversal Conditions

**A reply crosses sex or label**: the guard reports it, and waves stop preserving the plan.

**Sustained 429s**: the `retry` lines show them, and the in-flight default or the approach changes.

**A concurrent ceiling stop differs from the in-flight 1 stop**:
`test_a_mid_wave_ceiling_stops_at_the_in_flight_1_call` fails, and the reservation changes.

## Related

- [Call-log replay](call-log-replay.md)
- [E1's evaluation](e1-evaluation-protocol.md)
- [Record generation](record-generation-design.md)
