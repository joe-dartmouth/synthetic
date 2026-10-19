.NOTPARALLEL:
.PHONY: setup sync lint format typecheck test sizes rebased rebased-final synced check cells synthea reproduce

# Install the extras the gate needs from the lock (ADR tooling/uv-ruff-ty says when an extra joins
# this line). `--locked` refuses a lock that disagrees with pyproject.toml rather than rewriting it;
# the interpreter flag is one of the pin's homes `tests/test_repo_gate.py` checks.
setup:
	uv sync --locked --python 3.14 --extra dev --extra data --extra synth --extra eval --extra ontology --extra llm

# Install the exact lock across every extra.
sync:
	uv sync --locked --all-extras

# Every step below runs with `--no-sync`: the gate measures the environment `make setup` built and
# never re-locks or installs, so `synced` has the last word on what is installed.

# Lint and formatting check; writes nothing.
lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

# Apply formatting, including Python fenced in markdown. Not part of the gate.
format:
	uv run --no-sync ruff format .

# Static type check.
typecheck:
	uv run --no-sync ty check

# Run the suite across all cores with coverage. `[tool.coverage.*]` in pyproject.toml holds the
# measured sources, branch coverage and the floor; `worksteal` keeps the wall clock stable when a
# few tests dominate.
test:
	uv run --no-sync pytest -n auto --dist worksteal --cov

# Refuse a tree that `origin/main` has moved past (ADR tooling/refuse-a-stale-trunk-ref).
rebased:
	uv run --no-sync python -m tools.check_rebased

# The same question at the end of `make check`, since the trunk can move while the tests run.
rebased-final:
	@echo "rebased-final: asking again whether the trunk moved while the gate ran"
	uv run --no-sync python -m tools.check_rebased

# Refuse an environment whose installed packages disagree with `uv.lock`.
synced:
	uv run --no-sync python -m tools.check_lock_drift

# Refuse a tracked Python file over its line cap (ADR tooling/file-size-cap).
sizes:
	uv run --no-sync python -m tools.check_file_size

# Fetch UCI Heart Disease into data/ if absent, verify it, and print the training split's
# age-sex subgroup-label cell counts as CSV (docs/DEVELOPMENT.md, Data). Not part of the gate.
# `@` keeps make from echoing the command into stdout, so `make cells > counts.csv` is pure CSV.
cells:
	@uv run --no-sync python -m bko.datasets cells

# Generate the pinned Synthea population into data/ if absent (needs a JDK; JAVA_HOME is
# honored), ablate the default cell, and print the cell counts before and after as CSV.
synthea:
	@uv run --no-sync python -m bko.datasets synthea

# Rebuild Santos et al.'s MONDO/DO fusion example from the pinned releases (fetched into data/ if
# absent) and print one CSV row per quantity the paper reports (docs/DEVELOPMENT.md, Fusion
# reproduction). Not part of the gate.
reproduce:
	@uv run --no-sync python -m bko.ontology reproduce

# The whole gate, ordered by what invalidates what and then by cost (docs/DEVELOPMENT.md).
check: rebased synced lint sizes typecheck test rebased-final
