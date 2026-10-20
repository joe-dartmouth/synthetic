# Cleanup backlog

Small owed work that no task unit carries: review findings judged outside the boundary of the
change that found them, costs that do not bite at today's sizes, and gaps a later reader will trip
over. Each entry names its blocker when it has one.

The research queue lives elsewhere: planned work is a task unit
([`proposal-index.md`](proposal-index.md)), and unscheduled research is in
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

- **`cleanup:evaluate/replica-arms-accept-controls`: a control named in `replica_arms` crashes
  `score`.** `Experiment.__post_init__` checks only that a replica arm is an arm
  (`src/bko/evaluate/experiment.py:81-84`), and `_method_arm` hands a control an empty `Request()`
  (`src/bko/evaluate/run.py:52`), so a control in `replica_arms` fits a classifier on no rows and
  `score` exits 1 with a traceback. Fitting on a control's replica rows, as `score` would, prints
  `ValueError: Found array with 0 sample(s) (shape=(0, 5)) while a minimum of 1 is required by
  StandardScaler.` *Filed 2026-10-19 by `proposal-bko-replicas`.*
  **Why deferred.** No configuration names a control there.
  **What discharges it:** `__post_init__` refuses a replica arm that is a method whose `adds_rows`
  is false, with a test.
  **Trigger:** the next change to `Experiment`'s validation.

## fusion

- **`cleanup:fusion/completeness-counts-lines-not-calls`: the closure command checks each cell's
  line count, not that each of its calls answered.** `bko.fusion.__main__._answer` refuses a cell
  holding fewer lines than its rows. `bko.generate.decode` keeps every decoded item past the asked
  count (`src/bko/generate/decode.py:69-70`), so one call can write more lines than it asked for.
  With `SETTINGS.per_call` below a cell's rows, an early call that over-delivers can make up for a
  later call that never ran, and a failed run's file then passes as an answer. Today
  `per_call=20` equals `UNDERREPRESENTED_BELOW`, so each requested cell is one call and the hole
  cannot open. Found by static trace in PR #18's review. *Filed 2026-10-19 by
  `proposal-fusion-repair`.*
  **Why deferred.** The candidates file records no call's asked count, so the fix needs a format
  change in `bko.generate.candidates` or a call-log reader. Neither is warranted while each cell is
  one call.
  **What discharges it:** the closure command refuses a cell where any call wrote fewer lines than
  it asked for.
  **Trigger:** a change that sets `SETTINGS.per_call` below `UNDERREPRESENTED_BELOW`.

## generate

- **`cleanup:generate/candidates-carry-no-run-identity`: a candidates file does not name the split,
  releases or guidance level it was generated against.** `python -m bko.fusion` re-derives the
  split and the request and logs the split fingerprint. It refuses only a file whose cells or line
  counts do not answer the request, so a file generated against another split or release with the
  same requested cells is read as if it answered this one. The run's record ties the two through
  the candidates sha256 that both commands log (ADR `research/gap-closure-design`, Consequences).
  Raised in PR #18's review. *Filed 2026-10-19 by `proposal-fusion-repair`.*
  **Why deferred.** It changes the candidates format, which `bko.generate.candidates` writes and
  reads, and the one completeness run so far was checked by digest.
  **What discharges it:** generate writes the split fingerprint, release versions and guidance
  level where the closure command reads them, and the closure command refuses a mismatch.
  **Trigger:** `proposal-guidance-ablation`, which reads several completeness runs.
