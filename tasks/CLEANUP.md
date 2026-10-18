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

- **`cleanup:evaluate/bko-arms-reproduce-from-log`: a BKO arm's rows cannot be redrawn from a
  seed.** `docs/adr/research/e1-evaluation-protocol.md:26` says "Must reproduce from seeds on one
  machine", but the Messages API takes no seed parameter (ADR `research/record-generation-design`,
  Consequences). *Filed 2026-10-18 by `proposal-llm-generation`.*
  **Why deferred.** E1's protocol belongs to `proposal-final-evaluation`.
  **What discharges it:** E1's protocol states how a BKO arm's judged statistics reproduce.
  **Trigger:** `proposal-final-evaluation`.
- **`cleanup:evaluate/generated-copies-of-test-records`: a generated copy of a holdout's test
  record joins training unseen.** A model may reproduce a public Cleveland record. If it
  reproduces a holdout's test record, that record joins training, and utility is then read on a
  record the classifier trained on. E1's privacy measure reads distance to the training rows only
  (`docs/adr/research/e1-evaluation-protocol.md:72-74`). *Filed 2026-10-18 by
  `proposal-llm-generation`.*
  **Why deferred.** E1's measures belong to `proposal-final-evaluation`.
  **What discharges it:** `bko.evaluate` counts added rows equal to a test record.
  **Trigger:** `proposal-final-evaluation`.
