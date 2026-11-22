---
id: evaluate-syntheval-heart-disease
title: Score UCI Heart Disease fills under SynthEval's metrics
status: ready
depends_on: --
blocks: evaluate-syntheval-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/syntheval-reading
---

# Score UCI Heart Disease fills under SynthEval's metrics

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. [SynthEval](https://doi.org/10.1007/s10618-024-01081-4) is a metric library for
tabular synthetic data, and [Lautrup et al.](https://arxiv.org/abs/2507.19700) report heart-disease
results under it, with MIT code at
[disjoint-synthetic-data-generation](https://github.com/notna07/disjoint-synthetic-data-generation),
read at `a07cd9c`. SynthEval 1.7.3 needs numpy 2.0.0 or later and scipy 1.14.1 or later (a pure
wheel, Python 3.9 or later), so it cannot share an environment with SDNist.

## Scope

**In.**

- A scratch-lock probe that SynthEval 1.7.3 resolves against the project's lock. If it does, it
  becomes a new extra under ADR `tooling/uv-ruff-ty`; if not, it runs in a hashed environment
  outside the lock, as for SDNist.
- A command that scores an arm's UCI Heart Disease fill under the SynthEval metrics Lautrup et al.
  report.
- The mapping from their `hd` files to our frame. Their files hold 242 and 61 rows: the 303
  Cleveland rows with `cp`, `slope` and `thal` recoded and `target` 1 where UCI's `num` is 0, so
  `target` is inverted against ours.
- Named runs of the free arms.
- The EVIDENCE Configuration, `not yet run`.
- A new ADR under `docs/adr/research/` that records the probe's outcome and the mapping.

**Out.**

- Any paid arm.
- A claim about their generators.

## Acceptance

- A test scores a tiny fixture fill.
- `make check` passes.
