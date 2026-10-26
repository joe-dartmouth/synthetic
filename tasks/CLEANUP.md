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
## tests

- **`cleanup:tests/proposal-index-witness`: two corpus witnesses name `tasks/proposal-index.md`,
  which leaves when the proposal set's last unit lands.** `tests/test_docs_citations.py:91` and
  `tests/test_adr_frontmatter.py:666` each hold `Path("tasks/proposal-index.md")` as the file
  that proves the `tasks/` root is walked, and `tests/test_companion_guard.py:62` reads `A landed
  set's index is deleted with its last unit`. The proposal set holds one unit,
  `proposal-guidance-ablation`, so the commit that lands it deletes the index and fails both
  witnesses. *Filed 2026-10-26 by the diff review of PR #32.*
  **Why deferred.** The witnesses are not in the boundary of a change that adds a second task set.
  **What discharges it:** each witness names an index that outlives the proposal set,
  `tasks/experiments-index.md`, or `tasks/CLEANUP.md`, which no set's landing deletes.
  **Trigger:** the landing of `proposal-guidance-ablation`, or the next change to either witness
  list.
