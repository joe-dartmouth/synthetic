"""Scratch checkouts that run the root conftest in a child pytest.

Shared by `tests/test_skip_policy.py` and `tests/test_paid_guard_sessions.py`, which plant a few
test modules beside a copy of the root conftest and judge what a separate pytest run reports. A
child run is the only way to observe what the conftest does at configure time, at collection and
under xdist, since this run's own conftest is already active.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from tests._corpus import REPO_ROOT


COPIED: Final = (
    "pyproject.toml",
    "tests/__init__.py",
    "tests/conftest.py",
)
"""The files the root conftest needs to import and run one test, repo-relative.

`pyproject.toml` carries the marker registration and the `addopts` the scratch run reads.
"""

SCRATCH_RUN_TIMEOUT_S: Final = 300
"""One scratch pytest run over a handful of one-line tests, in seconds.

Generous because the child pays interpreter start-up and the conftest's imports, and a timeout
would read as a guard failure.
"""

INHERITED_PREFIXES: Final = ("PYTEST_", "COV_CORE_", "COVERAGE_")
"""Environment prefixes the scratch run must not inherit from the run that spawned it.

`PYTEST_ADDOPTS` and `PYTEST_PLUGINS` change what it selects and loads, xdist sets
`PYTEST_XDIST_*` in its workers, and pytest-cov's `COV_CORE_*` or coverage's `COVERAGE_*` would
start measurement in a run passed `--no-cov`.
"""


def scratch_checkout(tmp_path: Path, name: str, planted: Mapping[str, str]) -> Path:
    """Build a minimal checkout in a directory called `name`, holding the planted test modules.

    Args:
        tmp_path: The case's scratch directory.
        name: The checkout directory's name.
        planted: Each planted module's text, keyed by its repo-relative path.

    Returns:
        The checkout's root.
    """
    root = tmp_path / name
    for relative in COPIED:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, root / relative)
    for relative, source in planted.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(source, encoding="utf-8")
    return root


def scratch_pytest(
    root: Path, *args: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run pytest in the copied checkout over the given options and files.

    Args:
        root: The checkout.
        *args: The test files, relative to it or absolute, and any options before them.
        env: Variables set in the child, applied after the inherited prefixes are dropped.

    Returns:
        The finished run, its output captured.
    """
    child = {
        key: value for key, value in os.environ.items() if not key.startswith(INHERITED_PREFIXES)
    }
    child.update(env or {})
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed argv
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", "-rs", *args],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        stdin=subprocess.DEVNULL,
        timeout=SCRATCH_RUN_TIMEOUT_S,
        check=False,
        env=child,
    )
