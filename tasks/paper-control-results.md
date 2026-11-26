---
id: paper-control-results
title: Read the matched development contrasts
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration, paper-baselines, paper-novelty
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Read the matched development contrasts

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

The matched contrast determines which claim merits a larger study.

## Scope

Run the registered controls on development holdouts. Report paired differences and uncertainty
against each empirical and statistical control, plus sample sizes, failed fills and provenance.
Relate findings to E9 and the existing
[RESEARCH section 11](../docs/RESEARCH.md#control-reading-rule), the E12/E13 reading rule.

**Findings from planning.**

- The runs are entry E19's
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e19-matched-knowledge-controls-on-the-diabetes-130-removal)),
  in `RUNS`: `e19-thin` and `e19-empty` on `diabetes`, slices `e19-thina` to `e19-thinj` and
  `e19-emptya` to `e19-emptyj`; `e19-supported-thin` and `e19-supported-empty` on
  `diabetes-supported`, slices `e19-supported-thina` to `e19-supported-thinj` and
  `e19-supported-emptya` to `e19-supported-emptyj`. Their logs hold a
  `pool-random holdout=%d fill=%s fields=%s` line per `bko-pool-random` fill attempt, an
  `eb holdout=%d fill=%s kappa=%s` line per `eb-marginals` fill and an
  `hb holdout=%d fill=%s label=%s column=%s tau2=%s tau3=%s evidence=%.6f iterations=%d` line per
  `hb-marginals` label and modelled column, and a
  `pool holdout=%d fill=%s method=%s key=%s own=%d banded=%d` line per pooled method's or
  `hb-marginals`' cell.
- It reads the primary and secondary contrasts as RESEARCH section 11's
  [matched-control reading](../docs/RESEARCH.md#matched-control-reading) and E19's
  Configuration fix them, and no K contrast at full removal (ADR
  [`research/no-mapping-ablation`](../docs/adr/research/no-mapping-ablation.md),
  Consequences).
- It first reads `real` minus `none` on the supported runs; where that contrast is below the
  margin, the supported runs' contrasts are descriptive.
- Once E12 has run (unit
  [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md)), it checks that
  E19's `bko-direct`, `none` and `real` scores equal E12's per holdout.
- It reads E6's removed-truth reading beside the downstream contrasts: each arm's added rows in
  the removed cell scored against the removal's dropped records (ADR
  [`research/removed-truth-reading`](../docs/adr/research/removed-truth-reading.md)). E6's
  readings are `runs/e6-52` (`e6-thin`, stores `runs/e6-25-rows` to `runs/e6-34-rows`) and
  `runs/e6-53` (`e6-empty`, stores `runs/e6-35-rows` to `runs/e6-44-rows`), in E6's Runs rows
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)). The reader's
  command, from the main checkout with a worktree detached at a `run/<id>` tag (`--runs-dir`
  and `--data-dir` name the main checkout's folders; without `--runs-dir` the reader looks for
  run folders under the worktree):

  ```
  mkdir -p runs/<run-id> && test ! -e runs/<run-id>/truth.csv && test ! -e runs/<run-id>/truth.log && env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=<worktree>/src .venv/bin/python -m bko.evaluate truth <run> --runs-dir runs --data-dir data $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/truth.csv 2> runs/<run-id>/truth.log </dev/null
  ```
- The decision is recorded in E19's Read against in [`docs/EVIDENCE.md`](../docs/EVIDENCE.md).

## Acceptance

Record evidence for benefit, equivalence, harm or unresolved uncertainty. If empirical controls
explain the gains, carry conditional sampling and its limits forward as the paper question.
Record the decision before expanding runs.
