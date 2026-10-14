---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# uv, ruff and ty, wrapped by a Makefile

## Context

Four people work in this repository on a seven-week schedule, each on a different pipeline stage.
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

- **`uv`** to lock and install. `uv.lock` is committed; `make setup` and `make sync` both run
  `uv sync --frozen`, so neither can rewrite the lock.
- **`ruff`** for linting and formatting, configured in `ruff.toml`, **pinned exactly** in the `dev`
  extra.
- **`ty`** as the type checker, configured in `ty.toml`, **pinned exactly**. `pyrightconfig.json`
  is kept for editor integration only.
- **`hatchling`** as the build backend.
- **Runtime dependencies as optional extras** (`data`, `ontology`, `llm`, `synth`, `eval`, `app`),
  declared and locked now. **An extra joins `make setup` when something in `src/`, `tests/` or
  `tools/` imports it, and not before.** `make sync` installs every extra.
- **A `Makefile`** wrapping the canonical commands: `setup`, `sync`, `lint`, `format`, `typecheck`,
  `test`, `sizes`, `synced`, `rebased`, `check`.

## Rationale

A lock installed with `--frozen` is the only form that reproduces the same environment on four
machines; a plain `uv sync` re-locks when `pyproject.toml` is newer than `uv.lock`, and
`uv pip install` resolves fresh. A floating linter version changes verdicts on an unchanged tree,
because a new release adds rules, so ruff and ty carry `==` pins and a bump is a deliberate commit.

Declaring and locking every extra now proves the whole stack resolves on the chosen interpreter
before any stage is built, while installing only `dev` keeps `make setup` from downloading `torch`
for a teammate who is writing ontology code.

## Consequences

**Gained**: one command set across machines; a lock that proves the stack resolves; lint and type
verdicts that move only when a pin moves.
**Accepted**: a new lint rule never arrives on its own; a ruff or ty bump needs a commit and
`make setup`. A developer working on a stage installs its extra by hand until the stage's first
import moves it into `make setup`.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Poetry or PDM | Slower resolution and a second toolchain beside the reference repository's uv | 2026-10-13 |
| mypy as the primary checker | ty is part of the same Astral toolchain as uv and ruff, and the reference gate already runs on it | 2026-10-13 |
| Floating `>=` pins for ruff and ty | A release adds rules and changes the verdict on an unchanged tree | 2026-10-13 |
| Every runtime library in `dependencies` | A fresh clone would download `torch` and the whole stack before any stage imports it | 2026-10-13 |

## Enforcement

- Owner: `pyproject.toml`
- Pinned by: `tests/test_repo_gate.py`
- `test_repo_gate.py` asserts that ruff and ty carry exact pins, that the pins match the installed
  distributions, and that every declared dependency is in `uv.lock`.

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
