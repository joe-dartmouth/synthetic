.NOTPARALLEL:
.PHONY: setup sync lint format typecheck test sizes rebased rebased-final synced check cells synthea reproduce app video-env video syntheval-env syntheval-test

# Install the extras the gate needs from the lock (ADR tooling/uv-ruff-ty says when an extra joins
# this line). `--locked` refuses a lock that disagrees with pyproject.toml rather than rewriting it;
# the interpreter flag is one of the pin's homes `tests/test_repo_gate.py` checks.
setup:
	uv sync --locked --python 3.14 --extra dev --extra data --extra synth --extra eval --extra ontology --extra llm --extra app --extra slm

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

# Serve the Streamlit interface on 127.0.0.1:8501 until interrupted (docs/DEVELOPMENT.md, App).
# Not part of the gate; a live BKO fill spends money. The flags are the app's whole configuration
# (ADR research/streamlit-app-design).
app:
	uv run --no-sync python -m streamlit run src/bko/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --runner.fastReruns false --server.fileWatcherType none --client.toolbarMode minimal

# Build ManimGL's own environment in video/.venv from the hashed video/requirements.txt (ADR
# tooling/video-scenes-outside-the-lock). Not part of the gate; it installs outside the lock and
# leaves uv.lock and .venv untouched. The interpreter flag is video/.venv's, not one of the
# project pin's homes.
video-env:
	uv venv --allow-existing --python 3.14 video/.venv
	uv pip sync --python video/.venv/bin/python --require-hashes --only-binary :all: video/requirements.txt

# Render every scene of the demo video and join them into video/media/bko-demo.mp4
# (docs/DEVELOPMENT.md, Video). Not part of the gate; needs make video-env and FFmpeg.
video:
	@test -x video/.venv/bin/python || { echo "video: no video/.venv; run make video-env" >&2; exit 2; }
	@PYTHONPATH=src:. video/.venv/bin/python -m video.render

# Build SynthEval's own environment in bench/syntheval/.venv from the hashed
# bench/syntheval/requirements.txt (ADR tooling/benchmark-tools-outside-the-lock). Not part of the
# gate; it installs outside the lock and leaves uv.lock and .venv untouched. On macOS LightGBM,
# which SynthEval imports, needs Homebrew's libomp. The interpreter flag is the environment's, not
# one of the project pin's homes.
syntheval-env:
	uv venv --allow-existing --python 3.14 bench/syntheval/.venv
	uv pip sync --python bench/syntheval/.venv/bin/python --require-hashes --only-binary :all: bench/syntheval/requirements.txt

# Score a tiny fixture fill with the real SynthEval through bench/syntheval/score.py
# (docs/DEVELOPMENT.md, Benchmark environments). Not part of the gate; needs make syntheval-env.
syntheval-test:
	@test -x bench/syntheval/.venv/bin/python || { echo "syntheval-test: no bench/syntheval/.venv; run make syntheval-env" >&2; exit 2; }
	uv run --no-sync pytest bench/syntheval/tests -q -p no:cacheprovider --no-cov

# The whole gate, ordered by what invalidates what and then by cost (docs/DEVELOPMENT.md).
check: rebased synced lint sizes typecheck test rebased-final
