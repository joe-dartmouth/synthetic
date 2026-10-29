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

## evaluate

- **`cleanup:evaluate/fill-unnamed-in-arm-lines`: the guided, prescribed and baseline lines name
  no holdout or fill.** The direct arm's and the sampler's lines name both (`holdout=`, `fill=`,
  read from `bko.evaluate.fills.CURRENT_FILL`), but `guided arm=%s seed=%d outcome=...` and
  `prescribed arm=%s seed=%d wave=%d ...` (`bko.evaluate.guided`) and `filled method=%s drawn=%d
  added=%d` and `fit method=%s seconds=%.1f` (`bko.baselines.synthesizers`) carry neither, so a
  reader joins them to the `arm holdout=` line by order. *Filed 2026-10-29 by the diff review of
  PR #58.*
  **Why deferred.** The entries PR #58 paid named the direct arm's lines only.
  **What discharges it:** each of those lines names `holdout=` and `fill=` from `CURRENT_FILL`,
  documented in `docs/DEVELOPMENT.md` (Evaluation).
  **Trigger:** the next change to `bko.evaluate.guided`'s attempt line or
  `bko.baselines.synthesizers`' fill.

## generate

- **`cleanup:generate/flight-runner-after-settle-error`: a failure inside `_settle` leaves the
  flight on a closed runner.** In `bko.generate.flight.Flight._run`'s persistent-runner branch,
  `self._runner = asyncio.Runner()` runs after `self._settle(...)`, so if `_settle` raises (a
  `fail` or call-log write error), `self._runner` stays the runner `_settle` closed and the next
  `wave` raises `RuntimeError: Runner is closed` instead of the first error. *Filed 2026-10-29 by
  the diff review of PR #58.*
  **Why deferred.** It predates PR #58, which touched `_run` only to record what stopped the wave.
  **What discharges it:** the replacement runner is made in a `finally` around `_settle`, pinned by
  a test whose `fail` raises.
  **Trigger:** the next change to `Flight._run` or `Flight._settle`.

- **`cleanup:generate/sdk-debug-log-bodies`: SDK debug logging would write request bodies to
  stderr.** Nothing holds logger `anthropic._base_client` at INFO or above, so `ANTHROPIC_LOG=debug`
  makes the SDK log each request's options, prompt records included, to stderr, where
  `bko.generate.api.CallTag` tags them like any other record. Evidence: the SDK's `_setup_logging`
  sets that logger to DEBUG when `ANTHROPIC_LOG=debug`, and `HTTP_LOGGERS` lists only `httpx2`.
  *Filed 2026-10-29 by the diff review of PR #58.*
  **Why deferred.** It predates PR #58, and no documented command sets `ANTHROPIC_LOG`.
  **What discharges it:** `install_call_tag` or `configure_logging` holds that logger at INFO, or
  `docs/DEVELOPMENT.md` states what `ANTHROPIC_LOG=debug` writes.
  **Trigger:** the next change to `bko.generate.api`'s logging, or a run that sets
  `ANTHROPIC_LOG`.
