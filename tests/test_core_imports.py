"""`bko.core` imports no pipeline stage, so every stage can import it."""

from __future__ import annotations

import os
import subprocess
import sys

from tests._corpus import REPO_ROOT


def test_core_imports_no_stage() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    script = (
        "import sys, bko.core.logs\n"
        "print(sorted(m for m in sys.modules if m.startswith('bko.') "
        "and m != 'bko.core' and not m.startswith('bko.core.')))"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
