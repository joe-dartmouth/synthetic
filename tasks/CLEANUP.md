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

## docs

- **`cleanup:docs/e6-replays-unread`: E6's `e6-2` replay is not yet recorded.** E6 landed before
  the byte-for-byte replay of `e6-2` (`e6-thina`) at the run commit `22a58c9` finished, so its
  Runs row records `e6-12`'s replay and leaves `e6-2`'s to this entry
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)). It runs from the
  detached worktree `../synthetic-e6` into `runs/e6-2-replay/`; on 2026-11-04 `ls -l` showed its
  `scores.csv` at 0 bytes and `git worktree list` held `../synthetic-e6` at `22a58c9`. The
  worktree and the E6 power watcher (its pid in `scratchpad/e6-power.pid`) stay until it ends.
  *Filed 2026-11-04 by E6's landing.*
  **Why deferred.** The user chose to land E6 and move on rather than wait on the replays.
  **What discharges it:** the replay's `scores.csv` and `holdout-candidates.jsonl` compared with
  `cmp` against `runs/e6-2/`'s, the result recorded in E6's Runs row, the watcher stopped with
  `kill $(cat scratchpad/e6-power.pid)`, and the worktree removed with `git worktree remove` once
  `runs/` holds nothing under it.
  **Trigger:** the replay ending: its `scores.csv` non-empty and its call log ending in an `end`
  line.

## evaluate

- **`cleanup:evaluate/likelihood-curation-unused`: `Curation.LIKELIHOOD` and `over_ask` have no
  arm.** `bko.evaluate.experiment.MessagesArm` accepts `curation=Curation.LIKELIHOOD` with an
  `over_ask` above 1, and `tests/test_evaluate_experiment.py` pins both, but the curated arm they
  were reserved for does not run: G7's gate read `gate separated=False`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g7-bkb-likelihood-curation)), and the arm waits on
  the gate's reading on Diabetes 130. *Filed 2026-10-31 by PR7's landing.*
  **Blocked on** unit [`generate-bkb-curation-diabetes`](generate-bkb-curation-diabetes.md), which
  either runs them or finds the gate fails again.
  **Why deferred.** Removing them changes the Messages arm spec, outside PR7's docs-only landing.
  **What discharges it:** `Curation.LIKELIHOOD`, `over_ask` and their tests removed, with ADR
  `research/cllm-arm-design`'s spec bullet; or a scheduled curated arm that runs them.
  **Trigger:** unit `generate-bkb-curation-diabetes`'s phase 1 verdict.

- **`cleanup:evaluate/curator-scores-unkept`: a curated row's confidence and aleatoric scores are
  not kept.** `bko.evaluate.curation.classifier_discard` reduces each row to a boolean before
  `assess` reads it, so the candidates file says `dropped-curation` but not by what margin, and a
  threshold sweep needs a refit on the holdout's training rows. *Filed 2026-10-30 by the pre-run
  diff review of PR6.*
  **Why deferred.** Adding fields to `Seen` and the candidates schema after the review would move
  the run commit for a reading B3 does not need; the fit is deterministic (`random_state` 0, one
  thread), so a refit reproduces the scores.
  **What discharges it:** each curated candidate line carries its two scores, documented in
  `docs/DEVELOPMENT.md` (Evaluation), or a refit command that prints them.
  **Trigger:** a threshold sweep, or the next change to the candidates schema.

- **`cleanup:evaluate/log-time-offset`: log stamps carry no UTC offset.** Each stage's
  `configure_logging` (`bko.evaluate.__main__` and its peers) formats `%(asctime)s` as local time
  with no offset, so a `scores.log` stamp names no zone; `launch.log`'s stamps end in `Z`. The
  ambiguity bites across a clock change: `runs/g6-1-selfcheck/scores.log` begins
  `2026-11-01 13:37:13`, in EST, after commit `2a2a405` at `2026-11-01 13:36:55 -0500`, while the
  G6 runs' stamps of 2026-10-29 and 2026-10-30 are EDT. *Filed 2026-11-01 by G6's landing.*
  **Why deferred.** G6's landing is docs only, and the stamp format is shared by every stage's
  entry point.
  **What discharges it:** every `configure_logging` stamps ISO 8601 with its UTC offset (or UTC
  with `Z`), with a test, documented in `docs/DEVELOPMENT.md`.
  **Trigger:** the next change to a `configure_logging`.

- **`cleanup:evaluate/shared-capture-is-process-global`: `shared` captures the harness's log
  lines by changing a process-global logger.** `bko.evaluate.shared.captured` adds a handler to
  the `bko.evaluate.run` logger and lowers its level to INFO while a holdout is scored, then
  restores both, so two rescores in one process at once would read each other's `split` and `arm`
  lines, and a logger a caller disabled or stopped propagating surfaces as a check 3 or 4
  mismatch. The CLI rescores once per process. *Filed 2026-11-01 by G6's landing.*
  **Why deferred.** No caller rescores twice in one process or concurrently.
  **What discharges it:** `score_holdout` returns or yields the `split` and `arm` lines it logs (or
  takes a sink for them), and `rescore` checks those instead of a logger handler, with a test that
  runs two rescores in one process.
  **Trigger:** `rescore` called in-process twice or concurrently.

- **`cleanup:evaluate/removal-log-test-order`: a removal test fails after the CLI tests in one
  process.** `uv run --no-sync pytest tests/test_evaluate_cli.py tests/test_evaluate_removal.py -q
  -p no:cacheprovider --no-cov` fails
  `tests/test_evaluate_removal.py::test_each_holdout_logs_its_removal_and_flagged_cells` with a
  duplicate `removal` line: an earlier CLI test leaves a `bko` logger at INFO, and the test calls
  `prepared` again after its `caplog.at_level` block exits. `make test` spreads the files over
  workers and passes. *Filed 2026-11-01 by G6's landing.*
  **Why deferred.** The two files fail the same way in one process on `origin/main` at `2f5ab89`,
  so the failure sits outside G6's changes and its files.
  **What discharges it:** the test reads only the records its own `caplog.at_level` block
  captured, or the CLI tests restore the logger levels they set, with the two files passing in
  one process.
  **Trigger:** the next change to `tests/test_evaluate_removal.py` or to a CLI test's logging.

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

- **`cleanup:generate/anthropic-log-root-handler`: `ANTHROPIC_LOG` silences the bko INFO lines.**
  With `ANTHROPIC_LOG` set, importing the SDK runs its own `logging.basicConfig`, which installs a
  root handler first, so each command's `configure_logging` `basicConfig` does nothing and the
  root logger stays at WARNING: every `spend`, `guided` and `validity` line is lost. Evidence: the
  SDK's `anthropic/_utils/_logs.py` `_basic_config`. *Filed 2026-10-30 by the pre-run diff review
  of PR6.*
  **Why deferred.** It predates PR6, and no documented command sets `ANTHROPIC_LOG`
  (`docs/DEVELOPMENT.md` says never to set it to `debug`).
  **What discharges it:** each `configure_logging` passes `force=True`, pinned by a test that
  imports the SDK with `ANTHROPIC_LOG` set.
  **Trigger:** the next change to a `configure_logging`, or a run that sets `ANTHROPIC_LOG`.

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
