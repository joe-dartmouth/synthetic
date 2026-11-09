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

## app

- **`cleanup:app/gap-analysis-raw-split`: the app's gap analysis reads the raw training split,
  not the prepared one.** `bko.app.steps.analyze` counts cells and finds gaps on
  `split(...).train`, while its score step prepares each holdout through
  `bko.evaluate.run.score_holdout`, which applies the experiment's training fraction and removal
  (`bko.evaluate.prepare.prepared`). The app's experiment is `E1`, with neither, so the two agree
  today. *Filed 2026-10-31 by the diff review of PR #66.*
  **Why deferred.** No app run varies either today; changing the app's analysis is outside PR #66.
  **What discharges it:** `analyze` reads `prepared(...).ablated.train`, or the app refuses an
  experiment with a training fraction below 1 or a removal.
  **Trigger:** the app taking a named run other than `e1`.

## baselines

- **`cleanup:baselines/bayes-net-vectorize`: the Bayesian network draws row by row in Python.**
  `bko.baselines.bayes_net.draw` makes one `rng.choice` per row per field for the state and
  another for the donor; a `bayes-net --rows 200` command took 4.3 s for one 10000-row batch
  (measured by the diff review of PR #67). The learn side is paid: `learn` computes each (field,
  parents) mutual information once and `_node` counts by `np.bincount`, byte-identical to the
  recorded networks. *Filed 2026-10-31 by the diff review of PR #67.*
  **Why deferred.** Grouping the draw changes the seeded draw, so B2's recorded bytes would move.
  **What discharges it:** the draw grouped by parent configuration and by state, with donor arrays
  built once per field, with B2 rerun and recorded.
  **Trigger:** the next change that reruns B2 for another reason.

- **`cleanup:baselines/request-frame-defaults-to-uci`: the baselines fall back to UCI's frame.**
  `bko.baselines.request.Request.frame`, `bko.baselines.records.no_rows`, `training_set` and the
  other `bko.baselines.records` helpers, and `bko.baselines.synthesizers`' metadata default `frame`
  to `UCI_FRAME`, while the generate stage, the evaluate scoring helpers and the cell helpers
  require it. Every `bko.evaluate` caller passes the run's frame on the request. Evidence: `grep -rn
  "= UCI_FRAME" src/bko/baselines` lists them. *Filed 2026-10-29 by the diff review of PR #59.*
  **Why deferred.** The entries PR #59 paid named the generate stage and the cell helpers only.
  **What discharges it:** `frame` is required on `Request` and each helper, and ADR
  `research/uci-evaluation-frame` drops the baselines' exception.
  **Trigger:** the next change to `bko.baselines.request.Request`, or a non-UCI caller of a
  baseline outside `bko.evaluate`.

## evaluate

- **`cleanup:evaluate/frames-below-evaluate`: `bko.gaps` cannot import the Synthea and Diabetes
  frames.** The `contradictions` census is a gap-stage reading, but it lives in
  `bko.evaluate.contradictions`, with only its pure counters in `bko.gaps.contradictions`, because
  the Synthea frame and field rows (`SYNTHEA_FRAME`, `SYNTHEA_FIELD_ROWS` in
  `bko.evaluate.data`) and the Diabetes 130 ones (`DIABETES_FRAME`, `DIABETES_FIELD_ROWS` in
  `bko.evaluate.diabetes`) live in `bko.evaluate`, and `grep -n "^from bko.evaluate"
  src/bko/gaps/*.py` finds nothing: `bko.gaps` sits below `bko.evaluate`. *Filed 2026-11-05 by
  PR3 Phase 12 of the data-contradictions batch.*
  **Why deferred.** Moving the frames moves every evaluation import of them, outside the census's
  boundary, and the census reads them correctly from `bko.evaluate`.
  **What discharges it:** the Synthea and Diabetes 130 frames, indicators and field rows move to
  `bko.datasets`, and the census command moves to `bko.gaps`.
  **Trigger:** a module outside `bko.evaluate` other than `bko.app.steps` imports a frame loader;
  `grep -rn "^from bko.evaluate.\(data\|diabetes\) import" src/bko | grep -v "^src/bko/evaluate/"`
  lists only `src/bko/app/steps.py` until then.

## generate

- **`cleanup:generate/thinking-server-reasoning-check`: the server check does not verify a thinking
  id's reasoning mode.** `bko.generate.local.check_server` compares the `/props` alias and weights
  file name only, so a base-model server started under a `-thinking` alias without
  `--reasoning on --reasoning-format deepseek` passes it and the run measures thinking off under
  the thinking id; only `probe`'s `reasoning_chars=0` shows it. Evidence: `check_server`
  (`src/bko/generate/local.py`) and the PR #70 review (observability and security lenses). *Filed
  2026-11-02 by the thinking-cost landing.*
  **Why deferred.** The two thinking-cost runs are done and `probe` now exits 2 on a thinking id
  that returns no reasoning; a `/props` field for the reasoning setting needs a server build to
  check against.
  **What discharges it:** `check_server` reads the server's reasoning setting from `/props` (or
  refuses a thinking id when it cannot), pinned by a test whose server reports reasoning off.
  **Trigger:** the next thinking-on local run.

- **`cleanup:generate/thinking-reasoning-per-call-log`: no per-call reasoning share and no
  truncated-while-reasoning reason in the logs.** The `generate` line in `bko.generate.calls` has
  no reasoning split (`reasoning_chars` is logged only by `probe`), and a `max_tokens` stop after
  reasoning reads as `short ... reason=truncated` with no reasoning length, so thinking that ate
  the budget cannot be told from an answer cut short without `/tokenize` over the call log.
  Evidence: `src/bko/generate/calls.py` and the PR #70 review (observability lens). *Filed
  2026-11-02 by the thinking-cost landing.*
  **Why deferred.** The landed thinking-cost runs hold the split from `/tokenize` over their call
  logs, and the `generate` line's bytes are shared with every paid arm, so changing it is its own
  change.
  **What discharges it:** a local-only per-call line carries reasoning tokens and characters, and
  a truncated reply with reasoning logs its own reason, pinned by tests over a thinking reply.
  **Trigger:** the next thinking-on local run.

## tests

- **`cleanup:tests/root-guard-misses-warnings-and-quiet-levels`: the root-logger guard misses
  warning capture and quiet loggers' levels.** The guard in `tests/conftest.py` compares only the
  root's level and handlers. `bko.core.logs.configure` also calls `logging.captureWarnings(True)`,
  which swaps `warnings.showwarning`, and sets the levels of the loggers in `quiet`, so a test that
  calls `configure` can leak both unseen. Evidence: `src/bko/core/logs.py` and the PR #92 review
  (Python lens). *Filed 2026-11-08 by the review of PR #92.*
  **Why deferred.** Extending the guard may flag many existing tests, which is its own surface.
  **What discharges it:** the guard also snapshots and compares `warnings.showwarning` and the
  levels of the loggers `configure` quiets, and every test it then flags restores them.
  **Trigger:** the next change to the guard or to `bko.core.logs.configure`.
