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

- **`cleanup:evaluate/per-holdout-candidates-role`: no run-folder role holds the candidates a BKO
  arm generated on each holdout.** The paid-run protocol unit's scope gave `bko.app.runs.ROLES` and
  the Runs table of `docs/DEVELOPMENT.md` "a per-holdout candidates file the BKO arms generated";
  `ROLES` gained only `launch.log`, and `bko.evaluate` writes no such file, so a role would name an
  artifact nothing writes. `tasks/generate-bkb-curation.md` (Phase 1) reads that file. A named
  run's scores and criteria take the existing `scores.csv` and `criteria.csv` roles. *Filed
  2026-10-26 by the diff review of PR #37.*
  **Why deferred.** The writer does not exist; adding the role first breaks the rule that a role
  names what writes it.
  **What discharges it:** `score` writes each BKO arm's candidates per holdout into the run folder,
  `ROLES` and the Runs table gain the role, and `tests/test_run_folders.py` holds them equal.
  **Trigger:** the first change that writes per-holdout candidates, at the latest
  `generate-bkb-curation`.
