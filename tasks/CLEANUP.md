# Cleanup backlog

Small owed work that no task unit carries: review findings judged outside the boundary of the
change that found them, costs that do not bite at today's sizes, and gaps a later reader will trip
over. Each entry names its blocker when it has one.

The research queue lives elsewhere: planned work is a task unit on a set's index
([`proposal-index.md`](proposal-index.md), [`experiments-index.md`](experiments-index.md)), and
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

- **`cleanup:evaluate/spawn-child-reaped-early`: the spawn test can fail with no child to wait
  for.** One `make test` run on 2026-10-26 (under `pytest -n auto`) failed
  `test_spawn_starts_the_command_detached_with_its_output_in_the_files` in
  `tests/test_evaluate_launch.py` with these lines of its report:

  ```
  E       ChildProcessError: [Errno 10] No child processes
  tests/test_evaluate_launch.py:396: ChildProcessError
  ```

  The test passed alone and in the next full run. `bko.evaluate.launch.spawn` returns the pid of a
  `subprocess.Popen` it drops, and the test then calls `os.waitpid` on that pid. A likely cause,
  not yet confirmed: when the child has already exited, `Popen.__del__` polls and reaps it, so
  `os.waitpid` finds no child. *Filed 2026-10-26 by the `evaluate-frame-generic-pipeline` work.*
  **Why deferred.** It is unrelated to the change that met it, which touched no launch code.
  **What discharges it:** `spawn` keeps or returns the `Popen` (or the test waits through it), so
  nothing reaps the child before the waiter does, and the cause is confirmed.
  **Trigger:** the test's next failure, or the next change to `bko.evaluate.launch.spawn`.

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

- **`cleanup:evaluate/idle-timer-interrupt-escapes`: the idle-timer test can crash its xdist
  worker.** One `make check` run on 2026-10-27 (under `pytest -n auto`) failed
  `test_the_idle_timer_interrupts_a_blocked_main_thread` in `tests/test_evaluate_watch.py` with
  these lines of its report:

  ```
  worker 'gw13' crashed while running 'tests/test_evaluate_watch.py::test_the_idle_timer_interrupts_a_blocked_main_thread'
  FAILED tests/test_evaluate_watch.py::test_the_idle_timer_interrupts_a_blocked_main_thread
  ! xdist.dsession.Interrupted: <WorkerController gw13> received keyboard-interrupt !
  ```

  The file passed alone and the next two full runs passed. The test sets `idle_limit=0.05`.
  `bko.evaluate.watch.Watch.call` starts its timer before entering the `try` that turns the
  timer's interrupt into a `StopRuleError`. A likely cause, not yet confirmed: under load the
  timer fires in that window, and the interrupt escapes as a bare `KeyboardInterrupt` that ends
  the worker. *Filed 2026-10-27 by the `detect-synthea-mappings` work.*
  **Why deferred.** It is unrelated to the change that met it, which touched no evaluate code.
  **What discharges it:** `Watch.call` starts the timer inside its `try` (or otherwise catches an
  interrupt that fires before the call is entered), and the cause is confirmed.
  **Trigger:** the test's next failure, or the next change to `bko.evaluate.watch`.

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
  `bko.generate.prompt.system(frame)` reads the frame's columns, codes, units, ranges and decimals,
  but its opening sentence is fixed text about heart-disease prediction and patients referred for
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

- **`cleanup:generate/system-shim-for-pinned-test`: a shim serves one pinned test.**
  `bko.generate.prompt._system()` and its `FRACTIONAL` import exist only because the pinned
  `tests/test_generate_prompt.py` (lines 60-65) monkeypatches `FRACTIONAL` and calls `_system()`;
  the real path is `system(frame)`. *Filed 2026-10-27 by the review of PR #42.*
  **Why deferred.** The generate pin files stay unchanged through the frame refactor.
  **What discharges it:** the test calls `system(replace(UCI_FRAME, fractional=...))`, and the
  shim and the import are deleted.
  **Trigger:** the next change that may edit the generate pin files.
