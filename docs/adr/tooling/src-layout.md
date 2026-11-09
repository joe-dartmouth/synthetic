---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-11-08
---
# src layout, one subpackage per pipeline stage

## Context

The project's pipeline ([`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md)) has distinct stages
(Represent, Detect, Generate, Validate, Fuse, Evaluate, and a user interface) with different workstreams
and different heavy dependencies. Each teammate needs a home for their stage that exists before
any code does, so work can start in parallel without arguing about where files go.

## Decision Drivers

* **Must import the installed package, not the working directory**, so a packaging error fails
  locally.
* **Should give each stage a directory** with a stated role before the code exists.
* **Should map each stage to the one extra it will import.**

## Decision

We will use a **`src/` layout** with a single top-level package `bko` and one subpackage per
pipeline stage, plus `bko.core`, the helpers no stage owns:

| Subpackage | Stage | Extra it will import |
|---|---|---|
| `core` | Helpers no stage owns | none |
| `datasets` | UCI Heart Disease and Synthea loading | `data` |
| `ontology` | MONDO, HPO and DO loading; field mapping; BKO fragments | `ontology` |
| `bkb` | Bayesian Knowledge Base representation, fusion and reasoning | none |
| `gaps` | Ungrounded-statement and count-based gap detection | `data` |
| `generate` | Candidate-record generation through the Messages API or the direct sampler | `llm` |
| `validate` | Record consistency checks | `data` |
| `fusion` | Validated records joined back into the BKO; gap closure | `data` |
| `baselines` | No augmentation, class weighting, SMOTE, ADASYN, CTGAN, TVAE, the Gaussian copula, sequential CART | `synth` |
| `evaluate` | Utility, fidelity, privacy and validity metrics, and the BKO arms | `eval` |
| `app` | The Streamlit interface | `app` |

Every subpackage carries a `README.md` giving its role, its extra and its status; the
READMEs are excluded from the wheel. The package carries a `py.typed` marker, which the wheel ships
because it sits inside the package. Tests live in a top-level `tests/`.

## Rationale

With a flat layout the working directory shadows the installed package, so an import that only
works from the repository root passes every local run and fails for anyone installing the package.
The `src/` layout makes the test suite import what was installed.

One subpackage per stage gives each stage a boundary that matches the team's division of work,
and keeps a stage's heavy imports out of every other stage.

## Consequences

**Gained**: parallel work from day one; packaging errors surface locally; each stage's dependency
footprint is visible in its README.
**Accepted**: near-empty directories until the stages are built; code shared between stages
needs a deliberate home rather than an ad-hoc import across stages.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Flat layout (`bko/` at the root) | The working directory shadows the installed package | 2026-10-13 |
| One distribution per stage | A `pyproject.toml` and a lock per stage for one course project | 2026-10-13 |
| Notebooks as the primary code | Not importable, not linted, not testable by the gate | 2026-10-13 |
| A top-level `bko/logs.py` | It escapes the README and layout checks a subpackage gets | 2026-11-08 |

## Enforcement

- Owner: `src/bko/`
- Pinned by: `tests/test_package_layout.py`
- The test asserts that this table, `SUBPACKAGES` and the directories under `src/bko/` name the
  same stages; that every subpackage imports and its README carries the dependency and
  status lines; that each README's extra matches this table's; that every named extra exists in
  `pyproject.toml`; and that `py.typed` exists.

## Reversal Conditions

**A subpackage is added, merged or removed whenever the pipeline's stages change.** The table above
and `SUBPACKAGES` in `tests/test_package_layout.py` change in the same commit; the test fails if
they disagree.

**`bko.core` holds the stamped logging setup; a helper joins it only when two or more stages
import it and it belongs to neither.** A stage importing an upstream stage's public API (its
frames, constants, loaders, errors and data root, as `bko.baselines` imports `bko.datasets`) is a
pipeline dependency, not a shared helper.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
- [One home per kind of statement](one-home-per-kind-of-statement.md)
