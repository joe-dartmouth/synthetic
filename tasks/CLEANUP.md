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

- **`cleanup:evaluate/beside-flagged-scope-fingerprint`: `compare --beside` cannot tell whether
  two runs flagged the same test records.** `bko.evaluate.beside.pairable` pairs a holdout on its
  held-out summary scores and its test copies, and the `flagged-subgroups` scope is the test records
  of every age-sex subgroup holding a cell under the run's `fill_to` (`flagged` in
  `bko.evaluate.run.score_holdout`). Two runs whose `fill_to` or cell edges differ share held-out
  rows but flag different test records, and their flagged-scope series still pair. *Filed
  2026-10-26 by the review of PR #39.*
  **Why deferred.** Every registered run shares E1's `fill_to` and edges, and the scores file
  records no identity of the flagged cells.
  **What discharges it:** `score` records a per-holdout fingerprint of the flagged cells, and
  `pairable` compares it.
  **Trigger:** the first named run whose `fill_to` or cell edges differ from E1's.

- **`cleanup:evaluate/frame-unnamed-in-run-logs`: no log says which frame a run read.** No run
  log line, call log line, candidates header or `validated` summary names the frame, so two runs
  over different frames read alike. A refusal that is not a `GenerationError`, such as
  `CannedSend`'s frame refusal, reaches the call log as its type name only. Evidence:
  `bko.generate.log.CallLog`'s run line and `bko.evaluate.holdout_candidates`' header carry
  settings and origin with no frame field, and `bko.validate.checks._log` writes counts only.
  *Filed 2026-10-27 by the review of PR #42.*
  **Why deferred.** Naming the frame in the call log's run line changes call-log bytes, and only
  one frame runs.
  **What discharges it:** the run line, the candidates header and the `validated` line carry
  `frame=`, and `log.failure` keeps a refusal's message.
  **Trigger:** the first non-UCI frame that runs `score` or `generate`
  (`evaluate-synthea-downstream`).

- **`cleanup:evaluate/edges-default-to-uci`: helpers taking `edges` fall back to UCI's bands.**
  `bko.evaluate.experiment.flagged`, `bko.evaluate.run.held_out_ratio`,
  `bko.evaluate.utility.in_scope` and `utility`, and `bko.baselines.request.Strata.grid` and
  `stratum_keys` default `edges` to `AGE_BAND_EDGES`, so a caller holding a non-UCI frame that
  omits `edges` silently gets UCI's bands. *Filed 2026-10-27 by the review of PR #42.*
  **Why deferred.** Every caller in the harness passes its frame's edges, and only one frame runs.
  **What discharges it:** each takes the frame, or requires `edges`.
  **Trigger:** the first non-UCI frame that runs `score` or `generate`
  (`evaluate-synthea-downstream`).

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

- **`cleanup:evaluate/direct-training-copies-count`: no log counts added rows that equal a training
  row.** The direct arm copies training values by design, and nothing counts a drawn row that
  equals a training row in every column. Evidence: the smallest `bko-direct` `added` `dcr-p5` in
  `runs/g2-2/scores.csv` is 0.011978916663589707, above 0 on every holdout, which bounds such rows
  at 1 of 20 per holdout at most (inferred from the 5th percentile, not counted). *Filed 2026-10-27
  by the review of PR #47.*
  **Why deferred.** The bound needs no new run, and a count moves the recorded logs.
  **What discharges it:** `score` logs a zero-distance count of each arm's added and replica rows.
  **Trigger:** the next change to `bko.evaluate.direct`, or the next G2 run.

- **`cleanup:evaluate/holdout-candidates-schema-for-direct`: a direct run's candidates file reads
  as a Messages run's.** `bko.evaluate.holdout_candidates` lines now allow `call: null` under
  schema 1, and a run with no BKO arm still writes the Messages settings it never used into its
  header. Evidence, the start of `runs/g2-2/holdout-candidates.jsonl`'s header: `{"kind":
  "header", ..., "run": "g2", "schema": 1, "settings": {"effort": "medium", "guidance":
  "ontology", "max_tokens": 16000, "model": ...`. *Filed 2026-10-27 by the review of PR #47.*
  **Why deferred.** Bumping the schema moves the recorded bytes of `runs/g2-2`.
  **What discharges it:** the schema is bumped for a null call, or a run with no BKO arm writes no
  settings.
  **Trigger:** the next change to `bko.evaluate.holdout_candidates`.

- **`cleanup:evaluate/compare-header-run-level`: `compare`'s judged output names no run and no
  level.** The header log line, the `criterion=` lines and `criteria.csv` carry neither the run nor
  its guidance level, so two judged runs at different levels read alike; only the `--beside` lines
  name a run. Evidence: `bko.evaluate.compare` logs the header as `"compare candidate=%s dry_run=%s
  scores_sha256=%s"`, with the candidate arm, the dry-run flag and the scores' sha256 only. *Filed
  2026-10-28 by the diff review of PR #53.*
  **Why deferred.** `bko.evaluate.compare` lies outside the change that set `SETTINGS` to
  `summary`, which touches `SETTINGS` and its pins only.
  **What discharges it:** the header line carries `run=` and `level=`, pinned by a test.
  **Trigger:** the next change to `bko.evaluate.compare`.

- **`cleanup:evaluate/rehearsal-tokens-from-judged-run`: the rehearsal bills `e1-3`'s tokens per
  record, not the judged run's.** `INPUT_PER_RECORD` 368 and `OUTPUT_PER_RECORD` 197 in
  `bko.evaluate.rehearse` come from `e1-3`'s Runs row (34843870 and 18665554 over 94727 records
  asked); the judged run `e1-4` bills 300.535 input and 211.162 output tokens per record asked
  (`docs/EVIDENCE.md`, E1's Runs row). *Filed 2026-10-28 by the evidence review of PR #53.*
  **Why deferred.** The constants only scale a wiring check's spend and projection, and
  `bko.evaluate.rehearse` is code outside the evidence review's documents.
  **What discharges it:** the two constants come from the judged run `e1-4`, and the rehearsal's
  comments and ADR `research/e1-evaluation-protocol` name that run.
  **Trigger:** the next change to `bko.evaluate.rehearse`.

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

- **`cleanup:generate/system-prompt-uci-only`: the system prompt is UCI's for every frame.**
  `bko.generate.prompt.system(frame)` reads the frame's columns, codes, units and decimals, but
  its opening sentence is fixed text about heart-disease prediction and patients referred for
  coronary angiography, which `FrameSpec` has no field to replace. Its closing sentence indexes the
  frame's fractional fields, so a frame with none raises `IndexError` in
  `_alternatives` (`items[0]` of an empty list). Evidence: `system(TOY)` in
  `tests/test_datasets_frame.py` still opens with the heart-disease sentence. *Filed 2026-10-26 by
  the `evaluate-frame-generic-pipeline` work.*
  **Why deferred.** UCI is the only frame that generates records, and its prompt must stay
  byte-identical through that unit.
  **What discharges it:** the frame (or the caller) supplies the opening sentence, and a frame
  with no fractional field renders a closing sentence without one; UCI's text stays unchanged.
  **Trigger:** the first non-UCI frame that generates records (`evaluate-synthea-downstream`).

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
