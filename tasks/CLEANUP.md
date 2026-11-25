# Cleanup backlog

Small owed work that no task unit carries: review findings judged outside the boundary of the
change that found them, costs that do not bite at today's sizes, and gaps a later reader will trip
over. Each entry names its blocker when it has one.

The research queue lives elsewhere: planned work is a task unit on a set's index
([`experiments-index.md`](experiments-index.md) or [`direct-index.md`](direct-index.md)), and
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

## datasets

- **`cleanup:datasets/agreement-sending-placeholder`: any non-blank `sending` text permits
  sending.** `permits_sending` (`src/bko/datasets/sources.py`) reads:

  ```
  return agreement is None or bool((agreement.sending or "").strip())
  ```

  so a placeholder such as `TBD` passes as a permission to send records to a third-party API.
  Deferred because the fix is code, outside the docs change that plans the direct set's dataset
  units, and no pinned `Source` carries an `Agreement` with `sending` text yet.

  **What discharges it:** `permits_sending` refuses a `sending` value that is not a recognisable
  permission (a placeholder included), with a test that a placeholder is refused.

  **Trigger:** the first dataset unit that pins a `Source` with an `Agreement`.

## slm

- **`cleanup:slm/share-run-keeps-dropped-rows`: `bko.slm` trains on a synthetic-share run's
  dropped rows.** `read_holdout` (`src/bko/slm/__main__.py`) builds an arm's training set as

  ```
  training = pd.concat([split.train, added])
  ```

  and under a synthetic share `ready.ablated.train` is the whole training split, the dropped rows
  included (`Prepared.dropped`). So an E18 fill is added beside the rows it replaces, and at
  share 1 the adapter trains on twice the split's rows, where `bko.evaluate` trains each
  classifier on the split less the dropped rows plus the fill. Deferred because the change that
  added the share made no `bko.slm` edit, and no LoRA run reads an E18 folder yet.

  **What discharges it:** `read_holdout` refuses a run with a synthetic share, or trains on the
  split less `Prepared.dropped` plus the fill, with a test on a share run's holdout.

  **Trigger:** unit `slm-lora-results`, or the first change that points `bko.slm` at a share run.

  **Blocked on** the LoRA unit, which owns `bko.slm`'s training sets. Tracked in
  [`slm-lora-results`](slm-lora-results.md).
