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

## bkb

- **`cleanup:bkb/huge-int-reliability-overflows`: an integer reliability beyond float range escapes
  `fuse`'s refusal.** `_check` in `src/bko/bkb/fusion.py` accepts an `int`, then calls
  `math.isfinite`, which converts it. `fuse(list(h2_fragments()), [10**400, 1])` raises
  `OverflowError: int too large to convert to float`, not the `FusionError` its docstring promises.
  *Filed 2026-10-17 by the review of PR #11.*

  **Why deferred.** `_check` predates PR #11, which changed only the normalization; no caller
  passes such a reliability.

  **What discharges it:** `_check` refuses the value with `FusionError`, with a `REFUSED` case in
  `tests/test_bkb_fusion.py`.

  **Trigger:** the next change to `_check` in `src/bko/bkb/fusion.py`.

- **`cleanup:bkb/component-total-order-unverified`: pybkb's modes sum a component's total in
  S-node order, and pybkb's order there is unchecked.** `_add` sums source weights in the order
  `supports` first meets each source, which is not fragment order when fragments share S-nodes.
  Fragments A, B, C each holding `X=x` with probabilities 0.5, 0.25, 0.5, at reliabilities 0.1,
  0.7, 0.3 under `Normalization.COMPONENT`, give B the prior `0.6363636363636362`; fragment order
  gives `0.6363636363636364`. pybkb's fixtures never share an S-node across interleaved
  fragments, so they cannot tell which order pybkb uses. *Filed 2026-10-17 by the review of PR
  #11.*

  **Why deferred.** The order predates PR #11, and settling it means reading or running pybkb
  outside the lock.

  **What discharges it:** pybkb's summation order read from its source at the pinned commit, `_add`
  matched to it, and a test on the case above.

  **Trigger:** the next export of pybkb fixtures, or the next change to `_add`.

## ontology

- **`cleanup:ontology/exit-three-means-two-things`: a caller cannot tell a broken release from a
  disagreeing row by exit code.** Exit 3 covers a release that contradicts its pin or is not
  readable OBO (stdout empty) and a row that disagrees with a sound release (stdout holds the
  report). *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** One code for both is what the plan chose, and `docs/DEVELOPMENT.md` documents
  it. Changing it is a decision, not a fix.

  **What discharges it:** a distinct code for a disagreeing row, documented in
  `docs/DEVELOPMENT.md` (Ontologies), or a recorded decision to keep one code.

  **Trigger:** the first script or stage that calls the check and branches on its outcome.
