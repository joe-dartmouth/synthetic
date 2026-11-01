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

- **`cleanup:app/shared-split-in-app-means`: the app averages the `shared-split` marker.**
  `Measure.SHARED_SPLIT` is a held-out row whose value is a split's fingerprint read as an integer,
  not a quantity. The browse view's `bko.app.views.arm_means` averages it with the other held-out
  rows of a `shared` output's `scores.csv`, such as `runs/g6-1-shared/scores.csv`, and `merge`
  joins it as any other score; no test reads it in either. *Filed 2026-11-01 by G6's landing.*
  **Why deferred.** No judged reading uses the mean: `compare` judges other measures, and the
  browse view only displays it.
  **What discharges it:** `arm_means` leaves `shared-split` out of its means, and a `merge` test
  shows the row passes through unchanged, each with a test.
  **Trigger:** the next change to the app's means or to `merge`.

## baselines

- **`cleanup:baselines/bayes-net-vectorize`: the Bayesian network draws and learns row by row in
  Python.** `bko.baselines.bayes_net.draw` makes one `rng.choice` per row per field for the state
  and another for the donor, and `mutual_information` and `_node` loop over rows; a `bayes-net
  --rows 200` command took 4.3 s for one 10000-row batch (measured by the diff review of PR #67),
  and B2's runs took about 20 s a holdout on UCI and Synthea. *Filed 2026-10-31 by the diff review
  of PR #67.*
  **Why deferred.** B2's sizes run in minutes, and vectorizing changes the seeded draw, so B2's
  recorded bytes would move.
  **What discharges it:** the draw grouped by parent configuration and by state, with donor arrays
  built once per field, and the counts by `np.bincount` over a mixed-radix index, with B2 rerun and
  recorded.
  **Trigger:** the Diabetes 130 run of unit `evaluate-diabetes-subgroup-removal`, whose frame is
  hundreds of times larger.

- **`cleanup:baselines/classical-fit-line-and-docstrings`: the classical synthesizers log no fit
  time, and three docstrings overstate them.** CTGAN and TVAE log `fit method=%s seconds=%.1f`
  (`bko.baselines.synthesizers`), but `bko.baselines.classical.gaussian_copula` and `cart` log no
  fit line, so a B1 log has no fit duration for either. `bko.baselines.classical`'s module
  docstring says CART runs "as synthpop runs it" without naming the one-hot departure from rpart's
  subset splits (ADR `research/baseline-augmentation-design` names it), and `CART_COMPLEXITY`'s
  says "as rpart reads it" though rpart scales `cp` by the root's risk. `bko.evaluate.named_runs`'
  module and `RUNS` docstrings omit `b1-uci` and `b1-synthea`. *Filed 2026-10-29 by the diff review
  of PR #60.*
  **Why deferred.** B1 ran at commit `3788fa4`; changing `src` after the run would make its merge
  carry code the run did not.
  **What discharges it:** both methods log a `fit` line in CTGAN's format, the three docstrings
  state the current behavior, and `docs/DEVELOPMENT.md` (Baselines) names the fit lines.
  **Trigger:** the next change to `bko.baselines.classical` or `bko.evaluate.named_runs`.

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

- **`cleanup:evaluate/merge-configuration-check`: `merge` says it refuses slices of other
  configuration, but it compares score keys alone.** `bko.evaluate.merge`'s docstring says slices
  scored under different code or configuration are refused; it compares each slice's (arm, source,
  measure, scope) keys and holdouts, so two slices run at a different `training_fraction` or
  removal fraction share keys and merge. *Filed 2026-10-31 by the diff review of PR #66.*
  **Why deferred.** No run of a training fraction or removal fraction is sliced; the check predates
  PR #66.
  **What discharges it:** `merge` compares each slice's `experiment` line (or a configuration
  digest the scores carry), or its docstring says what it checks.
  **Trigger:** the next sliced run, or the next change to `bko.evaluate.merge`.

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

- **`cleanup:evaluate/shown-draw-in-flight-test`: no test runs a CLLM arm above one call in
  flight.** `bko.evaluate.cllm` and ADR `research/cllm-arm-design` say every in-flight count shows
  the same records; `Shower.take` runs only while a wave is planned, before the flight sends, but
  `tests/test_evaluate_cllm.py` runs one call at a time. *Filed 2026-10-30 by the pre-run diff
  review of PR6.*
  **Why deferred.** The claim follows from where `take` is called, and the replay test checks each
  request against the log.
  **What discharges it:** a test that runs a several-call CLLM attempt at in-flight 1 and 4 and
  finds the same logged requests.
  **Trigger:** the next change to `GuidedArm._planned` or `bko.evaluate.cllm`.

- **`cleanup:evaluate/data-line-agreement`: the `data` line names no agreement.** The `data
  dataset=... records=... digest=...` line (`bko.evaluate.data.load`) does not say which agreement
  the records are held under, so a run's log cannot show what permitted its sends. *Filed
  2026-10-30 by the pre-run diff review of PR6.*
  **Why deferred.** Every pinned source is open data (`tests/test_evaluate_egress.py`).
  **What discharges it:** the line carries `agreement=` (`none` for open data), documented in
  `docs/DEVELOPMENT.md` (Evaluation).
  **Trigger:** the first source held under an agreement.

- **`cleanup:evaluate/cllm-budget-remeasure`: `CLLM_BUDGET` comes from a proxy pilot that B3's
  run undercut.** ADR `research/cllm-arm-design` sets the budget from the least keep share of a
  pilot over BKO records (Synthea's 0.1734, giving 9). B3's `cllm` on Synthea kept 0.5370 of the
  rows reaching its curator against the pilot's 0.8856, and under half in five fills and cells,
  the least replica `<65/female/1` at 0.0707 (1437 of 20314;
  `runs/b3-3/holdout-candidates.jsonl`, the rows kept or dropped for the quota over those plus the
  rows dropped by curation), so the ADR's reversal condition holds and its rule gives
  `ceil(1.5 / 0.0707)` = 22. *Filed 2026-10-31 by the diff review of PR #63.*
  **Why deferred.** B3 ran at 9; changing the constant now would make the merge carry a budget no
  run used.
  **What discharges it:** `CLLM_BUDGET` re-derived from shown-record keep shares (B3's
  `dropped-curation` counts), the ADR's Budget and The pilot updated, and the B3 estimate test
  moved with it.
  **Trigger:** the next run of a curated arm.

- **`cleanup:evaluate/replica-one-label-guard`: a partial replica fill can leave the replica one
  label.** A curated arm's partial replica fill whose given-up cells leave rows of one label, or
  no rows, reaches `bko.evaluate.run._replica_scores`, which fits the logistic regression on the
  replica alone through `bko.evaluate.classifier.probabilities`; scikit-learn's fit then raises a
  plain `ValueError` mid-holdout, which the harness does not retry. It is latent on `b3-3`, whose
  replica kept rows of the positive class on every holdout. *Filed 2026-10-31 by the diff review
  of PR #63.*
  **Why deferred.** No run of record reaches it, and the fix touches the scoring path `b3-3` ran.
  **What discharges it:** a replica with fewer than two labels is scored as undefined or refused
  with an `EvaluationError` before the fit, with a test, documented in `docs/DEVELOPMENT.md`
  (Evaluation).
  **Trigger:** the next curated run on a frame with fewer positive cells (Diabetes 130).

- **`cleanup:evaluate/curated-one-record-calls`: the per-cell cap bounds records, not calls.** A
  curated arm's cell budget caps the records a cell asks, but a cell short by one asks one record
  per call, each paying the full prompt and shown records: 7745 of `b3-3`'s 28438 calls asked for
  one record (the `asked` field of `runs/b3-3/calls.jsonl`'s `request` lines). *Filed 2026-10-31
  by the diff review of PR #63.*
  **Why deferred.** The cap ends the runaway `b3-2` showed, and `b3-3` finished under its ceiling.
  **What discharges it:** a minimum ask per call, or a per-attempt cap on calls or USD, with a
  test, documented in ADR `research/cllm-arm-design` (Partial fill).
  **Trigger:** the next curated run.

- **`cleanup:evaluate/score-candidates-sha`: `score` logs no sha256 of the candidates file it
  writes.** `bko.evaluate.scoring` logs the scores' digest (`scores rows=%d sha256=%s`) but none
  for `holdout-candidates.jsonl`: `grep -c candidates runs/g6-2/scores.log` prints 0. G6's Runs
  rows state each candidates file's sha256 from the file on disk, not from the run's own log, while
  `likelihood` already logs the file it reads (`candidates path=%s schema=%d sha256=%s`,
  `bko.evaluate.__main__`). *Filed 2026-11-01 by G6's landing.*
  **Why deferred.** G6's landing is docs only, and the change touches the scoring path every run
  of record ran.
  **What discharges it:** `score` logs the candidates file's path and sha256 after it closes the
  file, with a test, documented in `docs/DEVELOPMENT.md` (Evaluation).
  **Trigger:** the next change to `bko.evaluate.scoring`.

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
