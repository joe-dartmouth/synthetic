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

## ontology

- **`cleanup:ontology/no-real-release-test`: the gate never parses a real release.** Every loader
  and table test reads OBO text built in `tmp_path` (`tests/_ontology_fixtures.py`); the only check
  against the pinned MONDO and HPO files is a person running `python -m bko.ontology check`. A
  pronto upgrade that changes how a real release parses passes `make check`.
  *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** The gate reads no download by design (AGENTS.md, Gotchas), so the test has to
  run outside `make check`, and `tests/test_skip_policy.py` governs how a conditional test may skip.
  Choosing that mechanism is outside the change that found it.

  **What discharges it:** an opt-in test that loads both pinned releases from `data/` and asserts
  the table checks clean, with a documented command, inside the skip policy.

  **Trigger:** the next re-pin of either release, or the next change to `bko.ontology.obo` or the
  pronto lock.

- **`cleanup:ontology/row-unions-not-exhaustive`: `check` renders an unknown row kind as
  another.** `_describe` in `src/bko/ontology/check.py` treats any row that is neither a
  `FieldMapping` nor a `CodeMapping` as `out-of-scope`, and `_clause` treats any clause that is not
  `Codes` as `AtLeast`. A new member of `ConceptRow` or `Clause` type-checks and prints the wrong
  subject. *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** Both unions have exactly the members the code handles today.

  **What discharges it:** both functions dispatch with `match` and end in `assert_never`, so `ty`
  refuses a new member until it is handled.

  **Trigger:** the next change to `ConceptRow` or `Clause` in `src/bko/ontology/schema.py`.

- **`cleanup:ontology/disagreements-only-on-stdout`: stderr does not name a disagreeing row.**
  When a row disagrees, `python -m bko.ontology check` logs only `checked N rows: M disagree`. With
  stdout redirected to a file, as `docs/DEVELOPMENT.md` shows, the terminal names no row.
  *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** The CSV on stdout names every disagreement, so no information is lost.

  **What discharges it:** one WARNING per disagreeing row, naming its concept and problem.

  **Trigger:** the next change to `src/bko/ontology/__main__.py`.

- **`cleanup:ontology/exit-three-means-two-things`: a caller cannot tell a broken release from a
  disagreeing row by exit code.** Exit 3 covers a release that contradicts its pin or is not
  readable OBO (stdout empty) and a row that disagrees with a sound release (stdout holds the
  report). *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** One code for both is what the plan chose, and `docs/DEVELOPMENT.md` documents
  it. Changing it is a decision, not a fix.

  **What discharges it:** a distinct code for a disagreeing row, documented in
  `docs/DEVELOPMENT.md` (Ontologies), or a recorded decision to keep one code.

  **Trigger:** the first script or stage that calls the check and branches on its outcome.

- **`cleanup:ontology/parser-version-not-logged`: a run's stderr does not name the pronto
  version.** Each release is logged with its name, sha256, `data-version` and term counts, so the
  input bytes are identified; the parser that read them is not. A pronto upgrade that changes a
  parse leaves no trace in the log. *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** `uv.lock` fixes the version for a reader who has the commit.

  **What discharges it:** one INFO line with `pronto.__version__` beside the release lines.

  **Trigger:** the next change to the release log lines in `src/bko/ontology/__main__.py`.

- **`cleanup:ontology/exact-match-scan-per-row`: each SNOMED-CT row scans the whole release.**
  `_match_problem` in `src/bko/ontology/check.py` walks every MONDO term (63329 at release
  2026-10-06) once per `CodeMapping` row to find other holders of the same `skos:exactMatch`. At
  one row the whole check runs in seconds, most of it parsing. *Filed 2026-10-16 by the review of
  PR #9.*

  **Why deferred.** It costs nothing measurable at one row.

  **What discharges it:** an index from each exact-match object to its live holders, built once
  per release.

  **Trigger:** a second `CodeMapping` row, which E2's ablation targets may add.
