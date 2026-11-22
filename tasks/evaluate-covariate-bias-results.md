---
id: evaluate-covariate-bias-results
title: Run the covariate-bias fractions and read them beside Juwara's Table 2
status: blocked
depends_on: evaluate-covariate-bias-fractions
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the covariate-bias fractions and read them beside Juwara's Table 2

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Unit [`evaluate-covariate-bias-fractions`](evaluate-covariate-bias-fractions.md) lands the runs
and their Configuration. This unit runs them and records what they show beside the published
table. Every arm is free.

## Scope

**In.**

- Run the registered runs from a detached worktree with `score` and `rows`, as
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives them. No API call.
- The EVIDENCE entry's Runs, Result and Read against, beside Juwara et al.'s Table 2.

**Out.**

- Any change to an arm, a fraction or a reading.

## Acceptance

- The entry's status is `complete`.
- Each run's sha256 is recorded.
- `make check` passes.
