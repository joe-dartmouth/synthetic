# Development: the environment, the gate and its hazards

How this repository is built, checked and run. [`AGENTS.md`](../AGENTS.md) keeps the working
summary; this file is the full record behind it. The repository's own checkers are described in
[`tools/README.md`](../tools/README.md).

## Environment

Requires `uv`. The interpreter version is the one in `.python-version`, and `uv` downloads it when
the machine lacks it. `make setup` installs the `dev`, `data` and `synth` extras from the lock, and
`source .venv/bin/activate` enters the environment.

- **Runtime dependencies are optional extras, declared and locked**: `data` (numpy, pandas) and
  `synth` (sdv, imbalanced-learn, torch) are installed by `make setup`; `ontology` (owlready2,
  pronto), `llm` (anthropic), `eval` (scikit-learn, sdmetrics) and `app` (streamlit) are installed
  on demand. `synth` brings `torch` to every machine, and on Linux the lock adds the NVIDIA CUDA
  packages and `triton`. ADR [`tooling/uv-ruff-ty`](adr/tooling/uv-ruff-ty.md) states when an extra
  joins `make setup`. `make sync` installs every extra.
- **Install an extra the lock-faithful way**: copy the `setup` recipe out of the `Makefile` and
  append `--extra <name>`. `uv sync` is exact, so a bare `uv sync --extra <name>` removes the
  others. Never `uv pip install -e ".[...]"`: it resolves fresh rather than from the lock, and
  `make synced` refuses the result.
- **Change a dependency with `uv lock`, then `make setup`.** Both install targets run
  `uv sync --locked`, which refuses a lock that disagrees with `pyproject.toml` instead of
  rewriting it, and every gate step runs `uv run --no-sync`, so nothing but an explicit `uv lock`
  writes `uv.lock`. `tests/test_repo_gate.py` compares the two offline as well.
- **Build backends are pinned** in `[tool.uv] build-constraint-dependencies`, because uv resolves
  them outside the lock: hatchling for `bko`, and setuptools, wheel and Cython for owlready2, which
  ships only an sdist.

## Data

The evaluation data is downloaded or generated, never committed. It lives under `data/` at the
repository root, which `.gitignore` ignores, and only
[`bko.datasets`](../src/bko/datasets/README.md) writes there. Each download (UCI Heart Disease and
the Synthea jar) is pinned by size and sha256 in `bko.datasets.sources`. A download that differs
from its pin is refused; the commands delete and fetch again a snapshot on disk that fails its pin,
and the UCI loader refuses one. A generated population is checked against its own `run.json`: the
commands regenerate one whose CSVs or recorded settings differ.

Both commands run inside the project's virtual environment (`.venv`, built by `make setup`), write
their report to stdout as CSV and their logs to stderr. The module exits 0 when done, 1 on an
unexpected error (with its traceback), 2 when the command could not run, and 3 when the data
contradicts its pin or schema; `make` reports any failure as its own exit 2. To use another data
root or to see the module's own code, run it directly:
`uv run --no-sync python -m bko.datasets <cells|synthea> --data-dir <dir>`.

- **The cell counts**: `make cells` runs `python -m bko.datasets cells`. It fetches UCI Heart
  Disease if `data/` lacks it, verifies it, and writes the age-sex subgroup-label cell counts of
  the training split, the same bytes on every run. The underrepresented cells and the snapshot
  identity go to stderr.
- **The Synthea ablation**: `make synthea` runs `python -m bko.datasets synthea`. It needs a JDK,
  17 or newer: `JAVA_HOME` when set, else `java` on `PATH`, so `JAVA_HOME=<jdk home> make synthea`
  picks one. The first run fetches the pinned Synthea jar and generates the population
  `DEFAULT_RUN` in `bko.datasets.synthea_run` names; later runs reuse it. It removes the cases
  `DEFAULT_ABLATION` in `bko.datasets.ablation` names and writes `age_band`, `sex`, `target`,
  `before` and `after` for every cell.
- **Layout**: a download's snapshot is `data/<source>/<first 16 hex digits of its sha256>/`, for
  example `data/uci-heart-disease/b17cd273da9ce1ca/`, holding the download and `source.json`. A
  population is `data/synthea/<key>/`, the key naming the jar's sha256 prefix, the seeds, the size,
  the snapshot date and the state; it holds Synthea's `csv/`, its log `synthea.log` and `run.json`.
  A failed generation leaves its log at `data/synthea/<key>.log`.
- **Identity**: `source.json` holds `url`, `size`, `sha256` and `retrieved_at` (the download time,
  UTC). `run.json` holds `run` (the parameters), `config`, `jvm_options`, `environment`,
  `jar_sha256`, `java_version`, `started_at`, `finished_at` and, per CSV, `rows` and `sha256`. An
  experiment copies them into its configuration in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry).
- **The gate is offline.** No test reads `data/` or the network; each builds its input in
  `tmp_path`, and no test runs Java.

## Baselines

`bko.baselines` ([README](../src/bko/baselines/README.md)) builds each comparison method's training
set from the training split `bko.datasets` computes (ADR
[`research/baseline-augmentation-design`](adr/research/baseline-augmentation-design.md)). One
command per method, inside `.venv`:

    uv run --no-sync python -m bko.baselines <none|class-weight> [--data-dir <dir>]
    uv run --no-sync python -m bko.baselines <smote|adasyn|ctgan|tvae> --rows <n> [--strata cell|label] [--seed <s>] [--data-dir <dir>]

- **What it does**: it fetches UCI Heart Disease if the data root lacks it, splits it with
  `SPLIT_SEED`, water-fills `--rows` over the strata (cells unless `--strata label`), runs the
  method and writes the training set to stdout as CSV: `record_id` (negative for an added row), the
  13 predictors, `target` and `weight`. `--seed` defaults to `BASELINE_SEED` in
  `bko.baselines.request`.
- **What it logs**: stderr gets the split line `cells` prints, one `quota` line per stratum, a line
  with the method, seed, strata, real, synthetic and total row counts and the output's sha256, and
  the method's parameters. An experiment copies them into its configuration in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry); the commit's `uv.lock` gives the library
  versions.
- **Exit codes**: as in [Data](#data), with 2 also for a request a method cannot fill and 3 for a
  method whose output breaks the contract.
- **Reproducibility**: a seed gives the same bytes on one machine with the same lock; CTGAN and
  TVAE promise no more, so the logged sha256 is the record.

## Platforms

The gate is POSIX-only: macOS and Linux. The checkers shell out to `git`, and the Makefile recipes
assume a POSIX shell.

**Use an SSH remote.** The checkers run `git` with global config blanked, which also drops any
global credential helper. With an HTTPS remote, `make rebased` cannot authenticate its `ls-remote`
probe, so it passes only in its degraded form and never confirms the trunk is current.

## Make targets

| Target | What it runs | Notes |
|---|---|---|
| `setup` | `uv sync --locked` with the `dev`, `data` and `synth` extras and the pinned interpreter | Refuses a stale lock |
| `sync` | `uv sync --locked --all-extras` | The whole lock, every extra |
| `lint` | `ruff check .` then `ruff format --check .` | `ruff format` reaches Python fenced in markdown |
| `format` | `ruff format .` | Rewrites files, documentation fences included; not part of the gate |
| `typecheck` | `ty check` | `ty.toml` sets the platform and paths; `pyrightconfig.json` is for editors |
| `test` | `pytest -n auto --dist worksteal --cov` | Sources, branch coverage and `fail_under` live in `pyproject.toml` |
| `sizes` | `python -m tools.check_file_size` | ADR [`tooling/file-size-cap`](adr/tooling/file-size-cap.md) |
| `cells` | `python -m bko.datasets cells` | Not part of the gate; fetches UCI Heart Disease on its first run ([Data](#data)) |
| `synthea` | `python -m bko.datasets synthea` | Not part of the gate; needs a JDK, and fetches the Synthea jar and generates on its first run ([Data](#data)) |
| `synced` | `python -m tools.check_lock_drift` | Installed packages against `uv.lock` |
| `rebased` | `python -m tools.check_rebased` | Probes `origin` with `git ls-remote`; ADR [`tooling/refuse-a-stale-trunk-ref`](adr/tooling/refuse-a-stale-trunk-ref.md) |
| `rebased-final` | The same checker again | Runs last, because the trunk can move while the tests run |
| `check` | `rebased synced lint sizes typecheck test rebased-final` | `.NOTPARALLEL`, so the order is a guarantee |

Every step from `lint` down runs as `uv run --no-sync`: the gate measures the environment
`make setup` built and never re-locks or installs during a run.

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
  `uv run --no-sync pytest tests/test_file_size.py -q -p no:cacheprovider --no-cov </dev/null`. A
  run whose every selected test skips exits 1, so a selection that reaches nothing cannot pass;
  `--allow-all-skipped` opts out where that is the intent.
- Every test module that names the changed file, found with `grep -rlF '<basename>' tests/`.

## Gate hazards

- **Stage before the gate.** `make sizes`, the conflict-marker scan and the root-file check read
  `git ls-files`, so an untracked file is under no gate.
- **Do not edit the tree while the gate runs.** Coverage resolves line numbers at report time, so an
  edit fakes a coverage miss below the edit point.
- **Do not run two covered gates at once.** They consume each other's `.coverage.*` fragments.
  Check `pgrep -f "pytest -n auto"` first and delete leftover `.coverage.*` files.
- **Paid tests are opt-in.** A test marked `live` calls a paid API and is skipped unless `--live`
  is passed; no `-m` expression selects it.
- **One test reaches the network.** The `check_rebased` recipe case in `tests/test_git_runner.py`
  runs the real probe against `origin` and carries the `integration` marker;
  `-m "not integration"` deselects it.
- **Importing SDV writes outside the repository.** `sdv.single_table` creates SDV's user data
  directory (`platformdirs.user_data_dir("sdv", "sdv-dev")`) and copies its logger configuration
  there on import, so the baselines tests touch it. No test reads it.
