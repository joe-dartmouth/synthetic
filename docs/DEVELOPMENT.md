# Development: the environment, the gate and its hazards

How this repository is built, checked and run. [`AGENTS.md`](../AGENTS.md) keeps the working
summary; this file is the full record behind it. The repository's own checkers are described in
[`tools/README.md`](../tools/README.md).

## Environment

Requires `uv`. The interpreter version is the one in `.python-version`, and `uv` downloads it when
the machine lacks it. `make setup` installs the `dev` extra from the lock (`uv sync --frozen`), and
`source .venv/bin/activate` enters the environment.

- **Runtime dependencies are optional extras, declared and locked but installed on demand**: `data`
  (numpy, pandas), `ontology` (owlready2, pronto), `llm` (anthropic), `synth` (sdv,
  imbalanced-learn), `eval` (scikit-learn, sdmetrics) and `app` (streamlit). An extra enters
  `make setup` when something in `src/`, `tests/` or `tools/` imports it, and not before (ADR
  [`tooling/uv-ruff-ty`](adr/tooling/uv-ruff-ty.md)). `make sync` installs every extra.
- **Add an extra the lock-faithful way**: copy the `setup` recipe out of the `Makefile` and append
  `--extra <name>`. `uv sync` is exact, so a bare `uv sync --extra <name>` removes the others. Never
  `uv pip install -e ".[...]"`: it resolves fresh rather than from the lock, and `make synced`
  refuses the result.
- **Change a dependency with `uv lock`**, then `make setup`. `--frozen` on both install targets means
  neither can rewrite `uv.lock`; only an explicit `uv lock` does.
- **Never run the gate as uid 0**: root bypasses the permission checks some repository tests rely
  on.

## Platforms

The gate is POSIX-only: macOS and Linux. The checkers shell out to `git`, and the Makefile recipes
assume a POSIX shell.

## Make targets

| Target | What it runs | Notes |
|---|---|---|
| `setup` | `uv sync --frozen` with the `dev` extra and the pinned interpreter | Lock-faithful; never re-locks |
| `sync` | `uv sync --frozen --all-extras` | The whole lock, every extra |
| `lint` | `ruff check .` then `ruff format --check .` | `ruff format` reaches Python fenced in markdown |
| `format` | `ruff format .` | Rewrites documentation fences too |
| `typecheck` | `ty check` | `ty.toml` sets the platform and paths; `pyrightconfig.json` is for editors |
| `test` | `pytest -n auto --dist worksteal` with branch coverage over `src/bko` and `tools` | `fail_under` lives in `pyproject.toml` |
| `sizes` | `python -m tools.check_file_size` | ADR [`tooling/file-size-cap`](adr/tooling/file-size-cap.md) |
| `synced` | `python -m tools.check_lock_drift` | `--no-sync`, so the launcher cannot repair the drift it measures |
| `rebased` | `python -m tools.check_rebased` | Probes `origin` with `git ls-remote`; ADR [`tooling/refuse-a-stale-trunk-ref`](adr/tooling/refuse-a-stale-trunk-ref.md) |
| `rebased-final` | The same checker again | Runs last, because the trunk can move while the tests run |
| `check` | `rebased synced lint sizes typecheck test rebased-final` | `.NOTPARALLEL`, so the order is a guarantee |

The `check` order puts what invalidates other steps first: a rebase can move `uv.lock`, which
invalidates `synced`, and a drifted environment makes `lint` and `typecheck` report errors that
belong to the environment rather than the tree. `sizes` precedes `test` because it is the cheaper
of the two.

## No CI, no hooks

There is no CI workflow and no git hook. `make check` is the whole gate and a person runs it before
opening a pull request. Name in the PR whether the whole gate or a targeted subset stands behind
it.

A targeted run while working:

- `make rebased synced lint sizes`, always: each guards something a narrow test run cannot see.
- `make typecheck` for any `.py` change.
- The test modules for what changed, without coverage, for example
  `uv run pytest tests/test_file_size.py -q -p no:cacheprovider --no-cov </dev/null`. A run whose
  every selected test skips exits 1, so a selection that reaches nothing cannot pass;
  `--allow-all-skipped` opts out where that is the intent.
- Every test module that names the changed file, found with `grep -rlF '<basename>' tests/`.

## Gate hazards

- **Stage before the gate.** `make sizes`, the conflict-marker scan and the root-file check read
  `git ls-files`, so an untracked file is under no gate.
- **Do not edit the tree while the gate runs.** Coverage resolves line numbers at report time, so an
  edit fakes a coverage miss below the edit point.
- **Do not run two covered gates at once.** They consume each other's `.coverage.*` fragments.
  Check `pgrep -f "pytest -n auto"` first and delete leftover `.coverage.*` files.
- **Paid tests are opt-in.** A test marked `live` calls a paid API and runs only under `--live`;
  selecting it with `-m live` alone skips it.
