---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# Python 3.14, without synthcity

## Context

The proposal names synthcity, alongside scikit-learn and SDMetrics, for evaluation. The reference
repository whose structure this one follows runs on Python 3.14, where coverage's
`sys.monitoring` core measures branch coverage; on earlier interpreters coverage falls back to its
slower C tracer whenever `branch = true`.

Resolution was measured on 2026-10-13 with `uv pip compile` for `aarch64-apple-darwin`:

- **synthcity 0.2.12** (the newest release on PyPI, uploaded 2025-05-08) requires
  `torch<2.3,>=2.1` and `numpy<2.0,>=1.20`.
- **On Python 3.14 it does not resolve**: no `torchtext` release satisfies `decaf-synthetic-data`,
  which synthcity requires, on that interpreter.
- **On Python 3.12 it resolves** to torch 2.2.2 and numpy 1.26.4, and resolving it with `sdv` drags
  sdv back to 1.29.1.
- **Every other library the proposal names resolves on 3.14**: sdv 1.38.5, ctgan 0.12.1,
  sdmetrics 0.32.0, imbalanced-learn 0.14.2, owlready2 0.51, pronto 2.7.3, streamlit 1.65.0,
  anthropic 1.13.0, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4.

## Decision Drivers

* **Must run every library the pipeline imports in one environment.**
* **Should match the reference repository's interpreter and gate.**
* **Must cover the evaluation the proposal specifies**: utility, fidelity and privacy
  (distance to closest record).

## Decision

We will use **Python 3.14**, and **synthcity is not a dependency**. Evaluation uses scikit-learn
for utility, SDMetrics for fidelity, and SDMetrics' `DCRBaselineProtection` and
`DCROverfittingProtection` for distance to closest record.

The interpreter version has six homes, each stated once: `.python-version`, `requires-python` in
`pyproject.toml`, `target-version` in `ruff.toml`, `python-version` in `ty.toml`, `pythonVersion` in
`pyrightconfig.json`, and the `setup` recipe in the `Makefile`.

## Rationale

Keeping synthcity would hold the whole stack to torch 2.2 and numpy 1.x and an SDV release
nine minor versions old, for metrics SDMetrics and scikit-learn already provide. The proposal's
privacy criterion is distance to closest record, which SDMetrics implements; nothing in the
proposal requires a metric only synthcity has.

## Consequences

**Gained**: one current environment on the reference repository's interpreter; the fast coverage
core with branch coverage.
**Accepted**: a departure from the proposal's tool list, recorded here and in the research plan.
Any metric only synthcity offers would need reimplementing.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Python 3.12 with synthcity | Forces torch 2.2.2, numpy 1.26.4 and sdv 1.29.1 on the whole stack | 2026-10-13 |
| A separate 3.12 environment for synthcity only | Two environments, two locks and data passed between them, for metrics already available | 2026-10-13 |
| Python 3.13 | Gains nothing over 3.14 and loses the coverage core | 2026-10-13 |

## Enforcement

- Owner: `pyproject.toml`
- Pinned by: `tests/test_repo_gate.py`
- The test asserts that the six homes agree, that the running interpreter matches them, and that
  coverage resolved to the `sys.monitoring` core.

## Reversal Conditions

**synthcity returns when a release resolves on Python 3.14 alongside the current stack.** The check
is `uv pip compile` of `synthcity` and `sdv` with `--python-version 3.14`: a solution that keeps
numpy 2 reverses the dependency half of this decision.

**The interpreter moves when a library the pipeline imports stops supporting 3.14**, which
`uv lock` reports as an unsatisfiable resolution.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
