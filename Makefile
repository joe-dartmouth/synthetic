.NOTPARALLEL:
.PHONY: setup sync lint format typecheck test sizes rebased rebased-final synced check

# Install the extras the gate needs from the lock. An extra joins this line when something in
# `src/`, `tests/` or `tools/` imports it (ADR tooling/uv-ruff-ty). `--frozen` reads the lock and
# never re-locks; the interpreter flag is one of the pin's homes `tests/test_repo_gate.py` checks.
setup:
	uv sync --frozen --python 3.14 --extra dev

# Install the exact lock across every extra.
sync:
	uv sync --frozen --all-extras

# Lint and formatting check; writes nothing.
lint:
	uv run ruff check .
	uv run ruff format --check .

# Apply formatting, including Python fenced in markdown.
format:
	uv run ruff format .

# Static type check.
typecheck:
	uv run ty check

# Run the suite across all cores with line and branch coverage. `worksteal` keeps the wall clock
# stable when a few tests dominate. The floor lives in pyproject.toml.
test:
	uv run pytest -n auto --dist worksteal --cov=src/bko --cov=tools --cov-branch

# Refuse a tree that `origin/main` has moved past (ADR tooling/refuse-a-stale-trunk-ref).
rebased:
	uv run --no-sync python -m tools.check_rebased

# The same question at the end of `make check`, since the trunk can move while the tests run.
rebased-final:
	@echo "rebased-final: asking again whether the trunk moved while the gate ran"
	uv run --no-sync python -m tools.check_rebased

# Refuse an environment whose installed packages disagree with `uv.lock`. `--no-sync` stops the
# launcher from repairing the drift this step measures.
synced:
	uv run --no-sync python -m tools.check_lock_drift

# Refuse a tracked Python file over its line cap (ADR tooling/file-size-cap).
sizes:
	uv run python -m tools.check_file_size

# The whole gate, ordered by what invalidates what and then by cost (docs/DEVELOPMENT.md).
check: rebased synced lint sizes typecheck test rebased-final
