---
id: evaluate-confirmatory-registration
title: Register the confirmatory runs, endpoint and contrasts on the new datasets and tag the commit
status: blocked
depends_on: evaluate-diabetes-control-results, evaluate-synthea-direct-results, evaluate-trace-attribution, baselines-tabddpm, datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame, evaluate-composite-dev-results
blocks: evaluate-confirmatory-results
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Register the confirmatory runs, endpoint and contrasts on the new datasets and tag the commit

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The confirmatory run is evidence only if the method, the comparators, the criteria and the
readings are fixed before any score exists on the new datasets.

## Scope

**In.**

- Confirmatory `RUNS` entries per new dataset:
  - candidate from unit [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md);
  - comparators `METHOD_ARMS` (`bko.evaluate.experiment`), `bayes-net`, `tabddpm`,
    `label-bootstrap` and `prior-shift`;
  - ablations `bko-direct`, `bko-nomap`, `bko-logic`, and `bko` where the dataset's agreement
    permits sending records.
- The removed cells, picked by one logged run of the rule unit
  [`datasets-selection-rule`](datasets-selection-rule.md) records, on the training splits.
- In [`docs/RESEARCH.md`](../docs/RESEARCH.md) section 11:
  - the primary endpoint, taken from the branch units
    [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) and
    [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) recorded;
  - one primary contrast per dataset;
  - Holm's correction across the comparators and the two health datasets, with its registry entry
    in [`docs/literature.md`](../docs/literature.md);
  - the equivalence margin;
  - the holdout count, derived from E6's interval widths
    ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) Configurations.
- A signed tag.
- ADR [`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md) revised
  in place.

**Out.**

- Any score, rehearsal or fit on a confirmatory dataset.
- Any change after the tag.

## Acceptance

- A test constructs every confirmatory `NamedRun` (`bko.evaluate.named_runs`).
- The tag exists, and each Configuration names its commit.
- `make check` passes.
