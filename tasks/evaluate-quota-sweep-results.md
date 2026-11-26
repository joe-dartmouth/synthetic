---
id: evaluate-quota-sweep-results
title: Run the quota sweep and read representativeness and recall per count
status: blocked
depends_on: evaluate-diabetes-control-results
blocks: evaluate-composite-arm, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the quota sweep and read representativeness and recall per count

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Whether the recommended count makes a cell representative, and what it costs or buys in recall,
is an empirical question. `RUNS` holds the sweep, E21
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e21-share-based-quota-sweep-on-the-diabetes-130-removal);
ADR [`research/share-based-quota`](../docs/adr/research/share-based-quota.md)); this unit runs it.

## Scope

**In.**

- The sweep's runs and readings for every arm at every count. The arms are those of the E6
  family of [E12](../docs/EVIDENCE.md#e12-control-arms-on-the-diabetes-130-removal), all
  free; no paid Messages arm runs.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Counts not registered.

**Findings from planning.**

- The runs: `e21-<word>-thin` and `e21-<word>-empty` for each word `coverage`,
  `coverage-two`, `coverage-four`, `coverage-eight`, `expected-size` and `recommended`, naming
  the rules `coverage`, `coverage-x2`, `coverage-x4`, `coverage-x8`, `expected-size` and
  `recommended` (`bko.evaluate.runs_direct.E21_NAMES`), each sliced `a` to `j`, for
  example `e21-coverage-thina` to `e21-coverage-thinj`. Each is E12's run with
  `Experiment.quota_rule` set, so the fixed fill is read from `e12-thin` and `e12-empty` with
  `compare --beside`, and E12's runs are this unit's prerequisite. Privacy reference arm: `real`.
- Each holdout logs one
  `quota-rule holdout=%d fill=gap key=%s rule=%s count=%d coverage=%d expected=%d asked=%d`
  line per flagged cell in `scores.log`; the ADR states, for holdout 0, how far each count sits
  from the fixed fill (`expected-size` and `recommended` within 30 rows, the coverage rules
  further).
- The reading, over each run's `rows` stores, one `--rows` per slice:

  ```
  mkdir -p runs/<run-id> && test ! -e runs/<run-id>/representativeness.csv && test ! -e runs/<run-id>/representativeness.log && uv run --no-sync python -m bko.evaluate representativeness <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/representativeness.csv 2> runs/<run-id>/representativeness.log </dev/null
  ```

  Its CSV holds `holdout,arm,cell,rows,gap,concept,cell_share,real_share,recall`; `recall` is
  the arm's flagged-subgroup recall, removed-cell recall on E6's cell.

## Acceptance

- The entry is `complete`, with the representativeness measure and recall at each count per arm.
- `make check` passes.
