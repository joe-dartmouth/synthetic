# Cleanup backlog

Small owed work that no task unit carries: review findings judged outside the boundary of the
change that found them, costs that do not bite at today's sizes, and gaps a later reader will trip
over. Each entry names its blocker when it has one.

The research queue lives elsewhere: planned work is a task unit on a set's index
([`experiments-index.md`](experiments-index.md)), and
unscheduled research is in
[`docs/IDEAS.md`](../docs/IDEAS.md). An entry here is not on any plan. It waits for a change that
passes close enough to pay it.

## The contract

- **A review that defers a finding files it here.** A finding fixed in its own diff needs no
  record. One judged out of boundary gets an entry, or it is lost.
- **Read this file before scoping a change.** Pay any entry in the surface you are about to touch.
- **Keys are `cleanup:<topic>/<slug>`.** The topic is the surface whose code must change, not the
  place the symptom shows: a stage under `src/bko/` (`ontology`, `datasets`, ...) or one of `docs`,
  `tests`, `tools`, `tasks`. The slug says what is wrong. Cite an entry by its key in a code span.
  Two filings of one finding write the same key and collide, which is the point. A key is never
  reused.
- **An entry carries its evidence**: a reproduction, a measurement, or the quoted output that
  shows the problem. It also says why it was deferred, what discharges it, and its trigger: the
  change that should pay it.
- **`Blocked on` marks an entry that cannot be paid yet.** It names the missing thing and links to
  where the unblocking work is tracked. A blocked entry is still owed; it is not a softer deletion.
- **An entry leaves when it stops describing owed work, and is deleted, not archived.** Git is the
  history; this file is the queue. Delete it in the commit that pays it, or in a commit that says
  why it no longer holds (it did not reproduce, its subject was removed, or another document now
  owns the work and states the argument itself). An entry paid in part stays, narrowed to what is
  still owed. Deleting an entry also deletes every citation of its key.
- **One heading per topic**, and a heading goes with its last entry.
- **No counts.** Derive them from the file when you need them.

`tests/test_cleanup_backlog.py` enforces the key form, the topic set, unique keys, that every cited
key resolves, that each entry names what discharges it and its trigger, and that a blocked entry
links to where its blocker is tracked.

## datasets

- **`cleanup:datasets/edges-default-to-uci`: the data stage's cell helpers fall back to UCI's
  bands.** `bko.datasets.split.split`, `bko.datasets.cells.age_band_labels`, `cell_keys` and
  `cell_counts`, `bko.gaps.observe.record_cells` and `bko.fusion.join.join` default `edges` to
  `AGE_BAND_EDGES`, so a caller holding a non-UCI frame that omits `edges` silently gets UCI's
  bands. Evidence: each signature reads `edges: Sequence[int] = AGE_BAND_EDGES`. *Filed
  2026-10-28 by the frame-selection work, which made the evaluate stage's helpers require
  `edges`.*
  **Why deferred.** They serve UCI-only commands (`bko.gaps uci`, `bko.fusion`, `cells`), and
  many call sites in those commands and their tests omit `edges`.
  **What discharges it:** each takes the frame or requires `edges`.
  **Trigger:** the first non-UCI frame reaching `bko.gaps` or `bko.fusion`.

## evaluate

- **`cleanup:evaluate/direct-arm-log-attribution`: the direct arm's log lines name no holdout or
  fill.** `bko.evaluate.direct`'s attempt line and `bko.generate.direct`'s `direct key=` and
  `fallback` lines carry the seed and cell only, so a per-holdout or gap-versus-replica split of
  fallbacks comes only from `holdout-candidates.jsonl`. Evidence, the first lines of
  `runs/g2-2/scores.log` (2026-10-27):

  ```
  INFO bko.evaluate.direct: direct arm=bko-direct seed=0 outcome=returned asked=20 generated=20 passed=20 dropped_range=0 dropped_cell=0 dropped_quota=0 kept=20
  INFO bko.generate.direct: direct key=<55/female/1 rows=18 records=2 group_records=110
  INFO bko.generate.direct: fallback key=<55/female/1 field=ca draws=11
  ```

  *Filed 2026-10-27 by the review of PR #47.*
  **Why deferred.** Changing those lines moves the recorded digest of `runs/g2-2/scores.log`.
  **What discharges it:** the attempt line and the sampler's `direct key=` and `fallback` lines
  carry the holdout and the fill.
  **Trigger:** the next change to `bko.evaluate.direct`, or the next G2 run.

## gaps

- **`cleanup:gaps/bkb-errors-unlogged`: a `BKBError` escapes `python -m bko.gaps` as a
  traceback.** `bko.gaps.__main__.main` catches `DatasetError` and `OSError` only, so a
  `bko.bkb.errors.BKBError` (`QueryError`, `FusionError`) raised while grounding exits 1 with a
  traceback instead of a logged refusal. Evidence: `main`'s only handlers are `except DatasetError`
  and `except OSError`, and `QueryError` and `FusionError` subclass `BKBError`, which subclasses
  `Exception` alone (`src/bko/bkb/errors.py`). Whether the current mappings can raise it is
  unverified. *Filed 2026-10-26 by the review of PR #41.*
  **Why deferred.** It predates the change that found it and takes the same path in `synthea` and
  `uci` as in `first-flag`, so it lies outside that change.
  **What discharges it:** `main` maps `BKBError` to a logged refusal with a documented exit code.
  **Trigger:** the next change to `bko.gaps.__main__.main`, or the first mapping that raises it.

## generate

- **`cleanup:generate/sdk-retries-unattributed`: retries without an HTTP response name no call.**
  `Attempts.handle_async` in `bko.generate.api` logs `retry call=%d attempt=%d status=%s
  retry_after=%s` only when a response with a retryable status reaches the middleware. A
  connection error or client timeout that the SDK retries produces only the SDK's own log record,
  which carries no call number, so with several calls in flight a reader cannot tie it to a call.
  Evidence: the middleware has no exception path that logs (`api.py`, `_log_retry` is called after
  `call_next` returns). *Filed 2026-10-27 by the review of PR #49.*
  **Why deferred.** It needs a `logging.Filter` that tags SDK records with
  `bko.generate.messages.CALL_NUMBER`, outside the review's fixes.
  **What discharges it:** every SDK retry record carries the call number, pinned by a test.
  **Trigger:** the next change to `bko.generate.api`, or a paid run whose stderr shows untagged
  retries.

- **`cleanup:generate/landed-line-latency`: the landed line gives no latency.**
  `bko.generate.calls` logs `send call=` at dispatch and `generate call=` on landing with no
  elapsed seconds or calls in flight, so a stalled call shows only as a `send` line with no later
  `generate` line. *Filed 2026-10-27 by the review of PR #49.*
  **Why deferred.** No stop rule or test depends on it.
  **What discharges it:** the landed line carries `elapsed_s=` and `in_flight=`, documented in
  `docs/DEVELOPMENT.md` (Evaluation).
  **Trigger:** the next change to the `generate call=` line.

- **`cleanup:generate/sigint-error-text`: a real Ctrl-C logs calls as `CancelledError`.**
  Under `bko.generate.flight`, `asyncio.Runner`'s SIGINT handler cancels the run, so each call in
  flight gets an `error` line reading `CancelledError` where a run at in-flight 1 wrote
  `KeyboardInterrupt`; the `end` line still reads `interrupted`. Evidence: a SIGINT probe over a
  three-call wave wrote `error 1 CancelledError`, `error 2 CancelledError`, `error 3
  CancelledError`, `end None interrupted`. *Filed 2026-10-27 by the review of PR #49.*
  **Why deferred.** Nothing in `src/` reads the error text, and the end status is right.
  **What discharges it:** calls cancelled by an operator's interrupt record `KeyboardInterrupt`.
  **Trigger:** the next change to `bko.generate.flight`'s interrupt path.

- **`cleanup:generate/required-frame-on-send-path`: the send path falls back to UCI's frame.**
  `bko.generate.messages.Call`, `bko.generate.calls.request`, `ask` and `generate`,
  `bko.generate.flight`'s `wave` and `bko.generate.prompt.render` default `frame` to `UCI_FRAME`,
  so a caller holding the Synthea frame that omits `frame` sends a UCI prompt and schema for its
  cell. Every `bko.evaluate` call site passes the run's frame. Evidence: each signature reads
  `frame: FrameSpec = UCI_FRAME`; `tests/test_datasets_frame.py` asserts
  `Call(SETTINGS, 0, 1, "system", "user").frame is UCI_FRAME`; `bko.generate.__main__` calls
  `generate` without `frame`; and the generate-stage tests call `Call`, `render`, `ask`,
  `request`, `generate` and `wave` without it (list them with an `ast` scan for those calls with
  no `frame` keyword). *Filed 2026-10-28 by the review of PR #56.*
  **Why deferred.** Requiring it rewrites the generate stage's test call sites, outside the
  Synthea frame change, and no Synthea call omits it.
  **What discharges it:** `frame` is a required keyword on each of those, and the default's pin
  in `tests/test_datasets_frame.py` goes.
  **Trigger:** the next change to `bko.generate`'s send path, or a second non-UCI caller of it.
