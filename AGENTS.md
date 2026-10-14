# synthetic: Bayesian Knowledge-Driven Ontologies for Synthetic Data

ENGG 119 team project (Partiban Anathurai, Shirley Asangi, Joe Laws, Krish Sapru). It tests whether
medical knowledge, held as a Bayesian Knowledge-driven Ontology (BKO), can find coverage gaps in a
clinical tabular dataset and guide synthetic-record generation that fills them, improving
heart-disease prediction for underrepresented groups. The Python package is `bko`.

> Status: scaffold. The toolchain and gate run; every `src/bko/` subpackage is a docstring-only
> stub. The proposal 2-pager is the plan of record and the research brief is background; where they
> disagree, the 2-pager governs.

This file is the single home for agent guidance in this repository. `CLAUDE.md` and `GEMINI.md`
import it (ADR `tooling/agent-config-single-home`).

## Commands

| Command | Purpose |
|---|---|
| `make setup` | Install the `dev` extra from `uv.lock` (`uv sync --frozen`) |
| `make sync` | Install the exact lock with every extra |
| `make lint` / `make format` | `ruff check` + `ruff format --check` / apply formatting |
| `make typecheck` | `ty check` |
| `make test` | `pytest` across all cores, with line and branch coverage |
| `make sizes` | Refuse a tracked Python file over its line cap (`CAPS` in `tools/check_file_size.py`) |
| `make synced` | Refuse an environment that has drifted from `uv.lock` |
| `make rebased` | Refuse a tree `origin/main` has moved past; probes the remote |
| `make check` | The whole gate: rebased, synced, lint, sizes, typecheck, test, rebased-final |

There is no CI and there are no git hooks: `make check` is the whole gate, and a person runs it.
Use targeted tests while working (`uv run pytest tests/<file> -q -p no:cacheprovider --no-cov
</dev/null`) and `make check` before opening a PR. [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) is
the full record of the environment, every target and the gate hazards.

## Layout

```
synthetic/
├── AGENTS.md / CLAUDE.md / GEMINI.md / README.md / Makefile / pyproject.toml / uv.lock
├── ruff.toml / ty.toml / pyrightconfig.json
├── docs/        # development guide and ADRs (docs/adr/)
├── src/bko/     # one subpackage per pipeline stage, each with a README
├── tools/       # the repository's own checkers, run by `make check` (tools/README.md)
└── tests/
```

## Gotchas

- **synthcity is not a dependency.** It pins `torch<2.3` and `numpy<2.0` and does not resolve on
  Python 3.14 (ADR `tooling/python-314-without-synthcity`). Evaluation uses SDMetrics and
  scikit-learn.
- **`make setup` installs `dev` only.** Runtime dependencies are declared and locked as extras
  (`data`, `ontology`, `llm`, `synth`, `eval`, `app`); an extra joins `make setup` when something in
  `src/`, `tests/` or `tools/` imports it (ADR `tooling/uv-ruff-ty`). To add one, copy the `setup`
  recipe out of the `Makefile` and append `--extra <name>`. Never `uv pip install`: it resolves past
  the lock and `make synced` refuses the result.
- **Stage new files before running the gate.** `make sizes` and several repository tests read
  `git ls-files`, so an untracked file is invisible to them.
- **`ruff format` formats Python inside markdown fences**, so an unformatted ```` ```python ````
  block in any doc fails `make lint`. Use a bare fence for a snippet that is not valid Python.
- **Never write a count, duration or other figure that moves** (test counts, file counts,
  runtimes). Cite the command that derives it instead; `uv run pytest --collect-only -q` prints the
  test count.

## Code style

- Python 3.14, 4-space indent, 100-character lines, double quotes (ruff-enforced).
- Google-style docstrings; full type annotations on every function and attribute.
- No `print()` in library code (use logging); no commented-out code.
- Absolute imports only.

## Documentation

- **One home per kind of statement** (ADR `tooling/one-home-per-kind-of-statement`). Point at the
  home rather than copying it.
- **State the current fact only.** Docs say what is true now, with its reason. No change-history
  narration; git holds history. `tests/test_no_history_narration.py` enforces this for this file,
  `CLAUDE.md`, `GEMINI.md`, `docs/DEVELOPMENT.md` and `tools/README.md`.
- **Prose describes guards; it never licenses a change** (ADR
  `tooling/prose-describes-guards-never-licenses-changes`). Run the guard or read its assertion.
- **ADRs have no index and no numbers.** Discover them with `docs/adr/**/*.md` and read their
  frontmatter; [`docs/adr/README.md`](docs/adr/README.md) holds the format. A record whose decision
  no longer exists is deleted, not archived.
- **A quotation mark means byte-for-byte.** Paraphrase without them.

**Document index:**

- Environment, platforms, Make targets, gate hazards: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)
- Decisions: [`docs/adr/`](docs/adr/README.md)
- The repository's own checkers: [`tools/README.md`](tools/README.md)
- Each pipeline stage: `src/bko/<stage>/README.md`

## Agent configuration

No `.codex/config.toml`: Codex's only project-level levers are the repo-wide `approval_policy` and
`sandbox_mode`, which are postures for each developer to choose rather than per-command grants.
