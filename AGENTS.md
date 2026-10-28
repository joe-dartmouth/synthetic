# synthetic: Bayesian Knowledge-Driven Ontologies for Synthetic Data

ENGG 119 team project (Partiban Anathurai, Shirley Asangi, Joe Laws, Krish Sapru). It tests whether
medical knowledge, held as a Bayesian Knowledge-driven Ontology (BKO), can find coverage gaps in a
clinical tabular dataset and guide synthetic-record generation that fills them, improving
heart-disease prediction for underrepresented groups. The Python package is `bko`.

> Status: every pipeline stage is built, and the gate runs; each `src/bko/` subpackage's README
> says what its stage holds. The plan of record is [`docs/RESEARCH.md`](docs/RESEARCH.md),
> converted from the team's [proposal 2-pager](papers/proposal-2-pager.pdf); the
> [research brief](papers/research-brief.pdf) is background, and where the two disagree the
> 2-pager governs (RESEARCH section 9).


This file is the single home for agent guidance in this repository. `CLAUDE.md` and `GEMINI.md`
import it (ADR `tooling/agent-config-single-home`).

## Commands

- `make setup`: install the `dev`, `data`, `synth`, `eval`, `ontology`, `llm` and `app` extras
  from `uv.lock`.
- `make check`: the whole gate. There is no CI and there are no git hooks, so a person runs it
  before opening a PR.
- `make format`: apply formatting.
- `make cells`: fetch UCI Heart Disease into `data/` and print its cell counts
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#data)).
- `make synthea`: generate the pinned Synthea population and print the default ablation's cell
  counts (needs a JDK).
- `make reproduce`: rebuild Santos et al.'s MONDO/DO fusion example from the pinned releases and
  compare each quantity with the paper
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#fusion-reproduction)).
- `make app`: serve the Streamlit app over UCI Heart Disease on `127.0.0.1`. A live BKO fill
  spends money and needs an Anthropic credential ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#app)).
- `uv run --no-sync python -m bko.baselines <method> [--rows N]`: build one comparison method's
  training set from the UCI training split and print it as CSV
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#baselines)).
- `uv run --no-sync python -m bko.evaluate score [--in-flight N] --log runs/<run-id>/calls.jsonl --candidates runs/<run-id>/holdout-candidates.jsonl > runs/<run-id>/scores.csv`,
  then `uv run --no-sync python -m bko.evaluate compare runs/<run-id>/scores.csv`: score every E1 arm
  on E1's holdouts, keeping every candidate a BKO arm generates per holdout, then judge `bko`
  against E1's criteria. `score` spends money through the three BKO arms and needs an Anthropic
  credential
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#evaluation) holds the guarded command and says what
  `--candidate` judges); `score --replay` rereads a paid run's call log for free, and
  `score --rehearse` runs every BKO call on canned replies for free. Each takes an optional run
  name first (default `e1`).
- `uv run --no-sync python -m bko.evaluate launch <run-id> [<run>] [--dry-run] [--in-flight N] [--rehearse | --resume <calls.jsonl>]`:
  start a new paid `score` run detached, kept awake and refused on battery power, into
  `runs/<run-id>/`, `<run>` defaulting to the run id's entry; it spends money.
  `--resume <calls.jsonl>` resumes a stopped run into the new folder under the same checks, and
  `--rehearse` rehearses the run on canned replies into `runs/<run-id>/rehearsal/` for free
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#evaluation) holds the paid-run protocol).
- `uv run --no-sync python -m bko.ontology check`: check the mapping table against the pinned
  MONDO and HPO releases, fetching them into `data/` if absent
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#ontologies)).
- `uv run --no-sync pytest tests/test_ontology_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null`:
  check the table and the fusion reproduction against the pinned releases already in `data/`
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#ontologies)).
- `uv run --no-sync python -m bko.gaps uci` or `uv run --no-sync python -m bko.gaps synthea`:
  list the gaps under each BKO gap definition and the count-based gaps of the UCI training split,
  or of the Synthea population before and after each of E2's ablations
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#gap-detection)).
- `uv run --no-sync python -m bko.gaps first-flag`: remove each whole E2 ablation's cases one at a
  time in its seed's draw order and list, per detector and truth item, the removed count at which
  the item is first flagged ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#gap-detection)).
- `uv run --no-sync python -m bko.generate per-subgroup --guidance ontology --log runs/<run-id>/calls.jsonl`:
  generate candidate records for E1's count-based gap request through the Anthropic Messages API,
  or for the completeness request with `--request completeness`. It spends money and needs an
  Anthropic credential ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#generation)).
- `uv run --no-sync python -m bko.fusion runs/<run-id>/candidates.jsonl`: read a completeness run's
  candidates, join the ones that pass validation to the training records of their cell, and list
  what closed, stayed open and is new under each BKO gap definition. It makes no API call
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#gap-closure) holds the guarded pair of commands).
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
├── runs/        # one folder per run of record, gitignored (docs/DEVELOPMENT.md#runs)
├── docs/        # research plan, background, architecture, roadmap, evidence, ADRs (index below)
├── papers/      # the source PDFs: the proposal 2-pager, the research brief and Santos et al. 2024
├── tasks/       # the open task sets: one unit per PR (tasks/<set>-index.md)
├── src/bko/     # one subpackage per pipeline stage, each with a README
├── tools/       # the repository's own checkers, run by `make check` (tools/README.md)
└── tests/
```

## Gotchas

- **synthcity is not a dependency.** It pins `torch<2.3` and `numpy<2.0` and does not resolve on
  Python 3.14 (ADR `tooling/python-314-without-synthcity`). Evaluation uses SDMetrics and
  scikit-learn.
- **pybkb is not a dependency.** It pins Python ~3.11 and numpy 1.24.4, so it does not resolve
  here. `bko.bkb` holds the part of it this project ports (ADR `research/bkb-layer-source`), and
  its test fixtures are exported from pybkb outside the lock
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#bkb-fixtures)).
- **`make setup` installs every declared extra: `dev`, `data`, `synth`, `eval`, `ontology`, `llm`
  and `app`.** ADR `tooling/uv-ruff-ty` says when a new extra joins `make setup`, and
  [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#environment) says how to try one before it does. Never
  `uv pip install`: it resolves past the lock and `make synced` refuses the result.
- **The gate never re-locks or installs.** Every gate step runs `uv run --no-sync`, and the install
  targets refuse a lock that disagrees with `pyproject.toml`. After editing a dependency, run
  `uv lock`, then `make setup`.
- **Downloaded data lives under the gitignored `data/`, and the gate never reads it.** Tests build
  their input in `tmp_path`, except the release tests, which run only with `--releases`
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#ontologies));
  [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#data) has the commands.
- **Paid tests need `--live`, and a session guard fails any other test that reaches the API.**
  `tests/conftest.py` refuses an Anthropic request made anywhere but a `live` test's body
  ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#gate-hazards)). Give a test a fake client instead.
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
  `CLAUDE.md`, `GEMINI.md`, `README.md`, `docs/DEVELOPMENT.md`, `tools/README.md` and every
  `src/**/README.md`.
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
- **Deferred work goes to [`tasks/CLEANUP.md`](tasks/CLEANUP.md).** A review finding left outside
  its change's boundary, or a small owed fix, is filed there under a `cleanup:<topic>/<slug>` key
  with its blocker, not left in a plan or a chat. Read it before scoping a change and pay what sits
  in the surface you touch. `tests/test_cleanup_backlog.py` enforces its contract.

**Document index:**

- Research plan, success criteria, datasets, source reconciliation:
  [`docs/RESEARCH.md`](docs/RESEARCH.md)
- Background, explained from scratch: [`docs/BACKGROUND.md`](docs/BACKGROUND.md)
- Architecture and pipeline: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Milestones and workload: [`docs/ROADMAP.md`](docs/ROADMAP.md); work units:
  [`tasks/experiments-index.md`](tasks/experiments-index.md)
- Results register: [`docs/EVIDENCE.md`](docs/EVIDENCE.md)
- Deferred cleanup and its blockers: [`tasks/CLEANUP.md`](tasks/CLEANUP.md)
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
