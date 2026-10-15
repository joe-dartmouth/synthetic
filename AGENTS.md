# synthetic: Bayesian Knowledge-Driven Ontologies for Synthetic Data

ENGG 119 team project (Partiban Anathurai, Shirley Asangi, Joe Laws, Krish Sapru). It tests whether
medical knowledge, held as a Bayesian Knowledge-driven Ontology (BKO), can find coverage gaps in a
clinical tabular dataset and guide synthetic-record generation that fills them, improving
heart-disease prediction for underrepresented groups. The Python package is `bko`.

> Status: scaffold. The toolchain and gate run; each `src/bko/` subpackage's README says whether
> its stage is built. The plan of record is [`docs/RESEARCH.md`](docs/RESEARCH.md), converted from
> the team's [proposal 2-pager](papers/proposal-2-pager.pdf); the
> [research brief](papers/research-brief.pdf) is background, and where the two disagree the
> 2-pager governs (RESEARCH section 9).

This file is the single home for agent guidance in this repository. `CLAUDE.md` and `GEMINI.md`
import it (ADR `tooling/agent-config-single-home`).

## Commands

- `make setup`: install the `dev`, `data` and `synth` extras from `uv.lock`.
- `make check`: the whole gate. There is no CI and there are no git hooks, so a person runs it
  before opening a PR.
- `make format`: apply formatting.
- `make cells`: fetch UCI Heart Disease into `data/` and print its cell counts
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#data)).
- `make synthea`: generate the pinned Synthea population and print the default ablation's cell
  counts (needs a JDK).
- `uv run --no-sync python -m bko.baselines <method> [--rows N]`: build one comparison method's
  training set from the UCI training split and print it as CSV
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#baselines)).
- A targeted run while working:

  ```
  uv run --no-sync pytest tests/<file> -q -p no:cacheprovider --no-cov </dev/null
  ```

[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#make-targets) holds every Make target, what the gate
runs, and its hazards.

## Layout

```
synthetic/
├── AGENTS.md / CLAUDE.md / GEMINI.md / README.md / LICENSE / Makefile / pyproject.toml / uv.lock
├── ruff.toml / ty.toml / pyrightconfig.json
├── data/        # downloaded datasets, gitignored, written by `bko.datasets`
├── docs/        # research plan, background, architecture, roadmap, evidence, ADRs (index below)
├── papers/      # the source PDFs: the proposal 2-pager and the research brief
├── tasks/       # the open task set: one unit per PR (tasks/proposal-index.md)
├── src/bko/     # one subpackage per pipeline stage, each with a README
├── tools/       # the repository's own checkers, run by `make check` (tools/README.md)
└── tests/
```

## Gotchas

- **synthcity is not a dependency.** It pins `torch<2.3` and `numpy<2.0` and does not resolve on
  Python 3.14 (ADR `tooling/python-314-without-synthcity`). Evaluation uses SDMetrics and
  scikit-learn.
- **`make setup` installs `dev`, `data` and `synth`.** The other runtime dependencies are locked
  extras installed on demand; ADR `tooling/uv-ruff-ty` says when one joins `make setup`, and
  [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#environment) says how to install one. Never
  `uv pip install`: it resolves past the lock and `make synced` refuses the result.
- **The gate never re-locks or installs.** Every gate step runs `uv run --no-sync`, and the install
  targets refuse a lock that disagrees with `pyproject.toml`. After editing a dependency, run
  `uv lock`, then `make setup`.
- **Downloaded data lives under the gitignored `data/`, and the gate never reads it.** Tests build
  their input in `tmp_path`; [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#data) has the commands.
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
- **Cite works through the registry.** [`docs/literature.md`](docs/literature.md) holds every work;
  `docs/RESEARCH.md` and `docs/IDEAS.md` cite it reference-style. `tests/test_docs_citations.py`
  checks citations, relative links and anchors across the docs, and that a link reading "section N"
  points at section N.
- **Task sets:** `tasks/<set>-index.md` is a set's graph and status board, and each unit is one PR.
  A landed unit's file is deleted after rehoming what it deferred; a set gets no companion page
  under `docs/`. `tests/test_task_units.py` enforces the unit schema.

**Document index:**

- Research plan, success criteria, datasets, source reconciliation:
  [`docs/RESEARCH.md`](docs/RESEARCH.md)
- Background, explained from scratch: [`docs/BACKGROUND.md`](docs/BACKGROUND.md)
- Planned architecture and pipeline: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Milestones and workload: [`docs/ROADMAP.md`](docs/ROADMAP.md); work units:
  [`tasks/proposal-index.md`](tasks/proposal-index.md)
- Results register: [`docs/EVIDENCE.md`](docs/EVIDENCE.md)
- Unscheduled ideas: [`docs/IDEAS.md`](docs/IDEAS.md); market and regulation:
  [`docs/MARKET.md`](docs/MARKET.md)
- Citation registry: [`docs/literature.md`](docs/literature.md); source PDFs: `papers/`
- Environment, data, platforms, Make targets, gate hazards:
  [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)
- Decisions: [`docs/adr/`](docs/adr/README.md)
- The repository's own checkers: [`tools/README.md`](tools/README.md)
- Each pipeline stage: `src/bko/<stage>/README.md`

## Agent configuration

No `.codex/config.toml`: Codex's only project-level levers are the repo-wide `approval_policy` and
`sandbox_mode`, which are postures for each developer to choose rather than per-command grants.
