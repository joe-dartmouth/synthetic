---
id: evaluate-removed-truth-reading
title: Read each fill against the removed records
status: ready
depends_on: --
blocks: paper-control-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/removed-truth-reading
---

# Read each fill against the removed records

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`paper-control-results` asks whether a fill's gain comes from the knowledge or from the records it
copies. A downstream score cannot say whether a fill resembles the records a removal took out of
a cell. The removal's dropped records are the truth of that cell, and each arm's added rows in the
cell can be scored against them with the fidelity measures a run already applies to a replica
(`src/bko/evaluate/run_scores.py`). The reading uses stores that already exist, so it costs no API
call and its first reading is recorded in the unit's own PR.

## Scope

**In.**

- `python -m bko.evaluate truth <run> --rows <store>...`, one `--rows` per slice store as
  `privacy` takes them, writing `truth.csv` and `truth.log`.
- Per holdout, the removal's dropped records re-derived with `bko.evaluate.prepare.prepared`,
  each arm's added rows in the removed cell taken from the store, and a score of those rows
  against the dropped records: the overall column-shape and pair-trend scores, and each column's
  shape score, as `src/bko/evaluate/run_scores.py` computes them for a replica.
- `real` left out: its rows are the dropped records, so its scores are 1.0.
- A run with no removal refused, and a store of another run refused.
- `src/bko/evaluate/truth.py`, its subcommand in `src/bko/evaluate/reader_commands.py` and
  `src/bko/evaluate/__main__.py`, and its entry in `ROLES` (`src/bko/app/runs.py`).
- `src/bko/evaluate/reader_commands.py` takes the `_shared`, `_rows`, `_panel`, `_privacy`,
  `_contradictions`, `_runs_root` and `_named` functions out of `__main__.py` unchanged, so
  `__main__.py` has room for the new subcommand.
- A new ADR under `docs/adr/research/` stating the measure, why the replica's fidelity measures
  are reused, and the Ruled Out readings: a new distance, and scoring against the retained cell.
- `docs/DEVELOPMENT.md`: the Evaluation paragraph and the Runs table rows for `truth.csv` and
  `truth.log`; `AGENTS.md`: a Commands entry in the form of the `privacy` entry;
  `src/bko/evaluate/README.md`.
- The first readings, free, of E6's `thin` and `empty` fractions over the `rows` stores that
  exist, recorded as two Runs rows of E6 in [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) with the
  command, the `run/<id>` tag and each file's sha256.
- `paper-control-results` drops this unit and becomes `ready`, and its unit file names the
  reader's command.

**Out.**

- E9's readings: E9 has no `rows` store.
- A new distance or a new fidelity measure.
- Scoring a fill against the retained cell, or a downstream score of any arm.
- Any API call, and any change to an existing `RUNS` entry, a stored run folder or `bko-direct`'s
  draw.

## Acceptance

- `tests/test_evaluate_truth.py` passes with `test_identical_rows_score_one`,
  `test_a_shifted_column_scores_below_one_on_that_column` (the hand-computed complement of a total
  variation distance on a two-code column), `test_the_dropped_records_are_the_removals`,
  `test_a_run_with_no_removal_is_refused`, `test_a_store_of_another_run_is_refused` and
  `test_real_is_left_out`.
- `tests/test_evaluate_cli.py`, `tests/test_evaluate_shared.py`, `tests/test_evaluate_rows.py`,
  `tests/test_evaluate_panel.py`, `tests/test_evaluate_donor_attacks.py`,
  `tests/test_evaluate_coherent.py` and `tests/test_evaluate_capped.py` pass after the move to
  `reader_commands.py`, and `wc -l src/bko/evaluate/__main__.py` reports below 400.
- `tests/test_run_folders.py` passes with `truth.csv` and `truth.log` known to the run-folder
  check.
- The command run over E6's `thin` stores writes a non-empty `runs/<id>/truth.csv` and a
  `runs/<id>/truth.log`, and the same command with `empty` writes `runs/<id2>/truth.csv`, each
  exiting 0 with ten `--rows` per fraction, or the slices that have a store named in the Runs
  row.
- `docs/EVIDENCE.md`'s E6 entry holds both Runs rows, each with its sha256s and one Result
  sentence per arm.
- `tests/test_task_units.py` passes with `paper-control-results` `ready` and this unit's file, board row and
  order-list entry gone.
- `make check` passes.
