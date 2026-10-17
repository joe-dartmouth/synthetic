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

- **`cleanup:ontology/no-real-release-test`: the gate never parses a real release.** Every loader
  and table test reads OBO text built in `tmp_path` (`tests/_ontology_fixtures.py`); the only checks
  against the pinned MONDO, HPO and DO files are a person running `python -m bko.ontology check`
  and `python -m bko.ontology reproduce`. A pronto upgrade that changes how a real release parses
  passes `make check`.
  *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** The gate reads no download by design (AGENTS.md, Gotchas), so the test has to
  run outside `make check`, and `tests/test_skip_policy.py` governs how a conditional test may skip.
  Choosing that mechanism is outside the change that found it.

  **What discharges it:** an opt-in test that loads the pinned releases from `data/` and asserts
  the table checks clean and the reproduction differs nowhere, with a documented command, inside
  the skip policy.

  **Trigger:** the next re-pin of any of them, or the next change to `bko.ontology.obo` or the
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

- **`cleanup:ontology/exit-three-means-two-things`: a caller cannot tell a broken release from a
  disagreeing row by exit code.** Exit 3 covers a release that contradicts its pin or is not
  readable OBO (stdout empty) and a row that disagrees with a sound release (stdout holds the
  report). *Filed 2026-10-16 by the review of PR #9.*

  **Why deferred.** One code for both is what the plan chose, and `docs/DEVELOPMENT.md` documents
  it. Changing it is a decision, not a fix.

  **What discharges it:** a distinct code for a disagreeing row, documented in
  `docs/DEVELOPMENT.md` (Ontologies), or a recorded decision to keep one code.

  **Trigger:** the first script or stage that calls the check and branches on its outcome.

- **`cleanup:ontology/exact-match-scan-per-row`: each SNOMED-CT row scans the whole release.**
  `_match_problem` in `src/bko/ontology/check.py` walks every MONDO term (63329 at release
  2026-10-06) once per `CodeMapping` row to find other holders of the same `skos:exactMatch`. At
  one row the whole check runs in seconds, most of it parsing. *Filed 2026-10-16 by the review of
  PR #9.*

  **Why deferred.** It costs nothing measurable at one row.

  **What discharges it:** an index from each exact-match object to its live holders, built once
  per release.

  **Trigger:** a second `CodeMapping` row, which E2's ablation targets may add.

## tasks

- **`cleanup:tasks/gap-request-source-unnamed`: no downstream unit says which gap list sets the
  generation request.** `tasks/proposal-llm-generation.md` generates candidate records per gap, and
  `tasks/proposal-final-evaluation.md` has the BKO arm return the rows per cell the gap request
  asks for. A BKO gap is an ontology statement with no age-sex cell (ADR
  `research/dataset-fragment-encoding`, Consequences), and on UCI Heart Disease the BKO list is
  empty (`docs/EVIDENCE.md`, E2, Result), so neither unit's text says what a BKO-driven request is.
  *Filed 2026-10-17 by the review of PR #12.*

  **Why deferred.** The plan kept every downstream unit's text unchanged beyond its status, and the
  answer depends on whether BKO grounding becomes per-subgroup, which the team decides.

  **Blocked on:** the team's answer to the per-subgroup grounding question in
  [PR #12](https://github.com/joe-dartmouth/synthetic/pull/12).

  **What discharges it:** both units name the gap list that sets the request and what the request
  is when that list is empty.

  **Trigger:** the first change to `tasks/proposal-llm-generation.md` or
  `tasks/proposal-final-evaluation.md` after the team answers.

- **`cleanup:tasks/bko-arm-split-range-unhandled`: a validated candidate can break the added-row
  contract.** `bko.validate` accepts a value inside the mapping table's range, and `check` in
  `src/bko/baselines/records.py` refuses an added row outside the training split's minimum and
  maximum. Cleveland's `trestbps` runs 94 to 200 against a range of 10 to 250, so a candidate with
  `trestbps` 210 passes validation and stops an E1 run with `OutputContractError`, which `_fill` in
  `src/bko/evaluate/run.py` re-raises without a retry. `tasks/proposal-final-evaluation.md` does
  not say how the BKO arm treats such a row. *Filed 2026-10-17 by the plan of
  proposal-record-validation.*

  **Why deferred.** The rule belongs to the unit that builds the BKO arm, and ADR
  `research/ontology-mapping-design` rules out data-derived bounds for the table.

  **What discharges it:** `tasks/proposal-final-evaluation.md` says how the BKO arm treats an
  accepted row outside the training split's range, and whether such a row counts against validity.

  **Trigger:** the first change to `tasks/proposal-final-evaluation.md`.

- **`cleanup:tasks/validity-denominator-unstated`: no unit says what reaches the validity
  denominator.** `bko.validate` counts every candidate it is given (ADR
  `research/record-validation-design`). `tasks/proposal-llm-generation.md` does not say that
  generation hands validation every decoded item, nor what it does with a response that does not
  decode. `tasks/proposal-final-evaluation.md` reads validity from `bko.validate`'s counts without
  saying they are summed over every call, including attempts `_fill` in `src/bko/evaluate/run.py`
  refuses and retries. Either gap removes failed records from the denominator and raises validity.
  *Filed 2026-10-17 by the plan of proposal-record-validation.*

  **Why deferred.** Both rules belong to the units that produce and consume the counts.

  **What discharges it:** `tasks/proposal-llm-generation.md` says every decoded item, and every
  response that does not decode, reaches `bko.validate`; `tasks/proposal-final-evaluation.md` says
  validity sums `passed` and `generated` over every call, refused attempts included.

  **Trigger:** the first change to `tasks/proposal-llm-generation.md` or
  `tasks/proposal-final-evaluation.md`.
