---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-21
---
# uv, ruff and ty, wrapped by a Makefile

## Context

Four people work in this repository, each on a different pipeline stage.
They need one set of commands that installs the same environment on every machine and returns the
same lint, type and test verdicts on the same tree. The repository follows the structure of the
author's reference research repository (mht3), whose toolchain already runs a strict gate on
Python 3.14.

The runtime stack is heavy: SDV brings `torch`, and the ontology, LLM, evaluation and UI stages each
add their own libraries. Most of it is imported by nothing until the stage that needs it is built.

## Decision Drivers

* **Must reproduce the locked environment exactly** on every teammate's machine.
* **Must give the same lint and type verdict for the same tree**, independent of when the
  environment was created.
* **Should keep a fresh clone light** until a stage actually needs its dependencies.

## Decision

We will use:

- **`uv`** to lock and install. `uv.lock` is committed. `make setup` and `make sync` run
  `uv sync --locked`, which refuses a lock that disagrees with `pyproject.toml`; every gate step
  runs `uv run --no-sync`, so the gate never re-locks or installs. Only `uv lock` writes the lock.
- **Build backends pinned** in `[tool.uv] build-constraint-dependencies`, since uv resolves them
  outside the lock.
- **`ruff`** for linting and formatting, configured in `ruff.toml`, **pinned exactly** in the `dev`
  extra.
- **`ty`** as the type checker, configured in `ty.toml`, **pinned exactly**. `pyrightconfig.json`
  is kept for editor integration only. **`pandas-stubs`**, which ty reads pandas through, is pinned
  exactly too, to the release that describes the locked pandas.
- **`hatchling`** as the build backend.
- **Runtime dependencies as optional extras** (`data`, `ontology`, `llm`, `synth`, `eval`, `app`),
  declared and locked now. **An extra joins `make setup` when something in `src/`, `tests/` or
  `tools/` imports it, and not before.** `make sync` installs every extra.
- **A `Makefile`** wrapping the canonical commands: `setup`, `sync`, `lint`, `format`, `typecheck`,
  `test`, `sizes`, `synced`, `rebased`, `rebased-final`, `check`, `cells`, `synthea` and
  `reproduce`, which run dataset and ontology commands outside the gate, and `app`, which serves
  the Streamlit interface outside it.

## Rationale

A lock installed as recorded is the only way to reproduce the same environment on four machines.
A plain `uv sync` or `uv run` re-locks and re-installs when `pyproject.toml` is ahead of `uv.lock`,
and `uv pip install` resolves fresh, so either can change the environment under a gate that has
already measured it. `--locked` turns a stale lock into an error at install time, and `--no-sync`
keeps every gate step from repairing what `synced` measures. A floating linter version changes verdicts on an unchanged tree,
because a new release adds rules, so ruff and ty carry `==` pins and a bump is a deliberate commit.

Declaring and locking every extra proves the whole stack resolves on the chosen interpreter before
any stage is built, while installing only `dev` and the extras something imports keeps
`make setup` from downloading the libraries of stages nothing imports yet.

## Consequences

**Gained**: one command set across machines; a lock that proves the stack resolves; lint and type
verdicts that move only when a pin moves.
**Accepted**: a new lint rule never arrives on its own; a ruff or ty bump needs a commit and
`make setup`. A developer working on a stage installs its extra by hand until the stage's first
import moves it into `make setup`. `make setup` installs `synth` because `bko.baselines` imports
it, `eval` because `bko.evaluate` imports it, `ontology` because `bko.ontology` imports pronto,
and `llm` because `bko.generate.api` imports anthropic, as does `tests/conftest.py` for the
paid-request guard, and `app` because `bko.app` imports streamlit, which `make typecheck` must
resolve. `synth`
brings `torch` to every clone, and on Linux the lock adds the NVIDIA CUDA packages and `triton`.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Poetry or PDM | Slower resolution and a second toolchain beside the reference repository's uv | 2026-10-13 |
| mypy as the primary checker | ty is part of the same Astral toolchain as uv and ruff, and the reference gate already runs on it | 2026-10-13 |
| Floating `>=` pins for ruff and ty | A release adds rules and changes the verdict on an unchanged tree | 2026-10-13 |
| Every runtime library in `dependencies` | A fresh clone would download `torch` and the whole stack before any stage imports it | 2026-10-13 |
| `uv sync --frozen` for the install targets | Installs a stale lock without a word when `pyproject.toml` has moved | 2026-10-14 |
| Bare `uv run` in gate steps | Re-locks and installs mid-gate after `synced` has measured the environment | 2026-10-14 |
| PEP 735 dependency groups for the dev tools | The reference repository uses an extra, and the gate's tests read extras; a group would split the convention for no gain in an unpublished package | 2026-10-14 |

## Enforcement

- Owner: `pyproject.toml`
- Pinned by: `tests/test_repo_gate.py`
- `test_repo_gate.py` asserts that ruff and ty carry exact pins, that the pins match the installed
  distributions, that `uv.lock` records exactly the requirements `pyproject.toml` declares, that
  `make setup` installs at least the `dev`, `data`, `synth`, `eval`, `ontology`, `llm` and `app`
  extras, and that every `uv run` recipe carries `--no-sync` and every `uv sync` recipe `--locked`.

## Reversal Conditions

**ty stops being the primary checker if `make typecheck` cannot give a verdict on the tree** and
pyright is needed for it; the two then swap roles. The check is what `make typecheck` runs, not how
young ty is.

**An extra joins `make setup` the first time something in `src/`, `tests/` or `tools/` imports one
of its packages.** A grep for the extra's top-level module names answers it. This is the mechanism
the decision records, not a reversal of it.

## Related

- [Python 3.14 without synthcity](python-314-without-synthcity.md)
- [src layout](src-layout.md)
