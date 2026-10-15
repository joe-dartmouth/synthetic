---
id: proposal-eval-harness
title: Evaluation harness
status: ready
depends_on: --
blocks: proposal-streamlit-app, proposal-final-evaluation
parallel_safe: true
owner: Joe
milestone: status-update-2
paid: false
adrs: --
---

# Evaluation harness

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The criteria need rare-class recall and AUC-PR, fidelity, privacy as distance to closest record, the
validity rate, and gap closure ([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria),
[section 5](../docs/RESEARCH.md#5-evaluation-contract)).

## Scope

**In.**

- Train-on-synthetic and train-on-real scoring with scikit-learn, SDMetrics fidelity, and SDMetrics'
  DCR metrics in `src/bko/evaluate/`.
- Fix the three open readings of the criteria
  ([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)) in the experiment configuration before any run.
- Fix in the experiment configuration the request each baseline is given and the reference rows for
  generator fidelity and train-on-synthetic scoring: a baseline fills exactly the rows per stratum it
  is asked for (ADR `research/baseline-augmentation-design`).

**Out.** Running the final comparison.

## Acceptance

- The harness scores every baseline end to end, class weighting's sample weights included.
- The `eval` extra joins `make setup`.
- `make check`.
