"""`bko.ontology.frames` imports no `bko.evaluate` module, so `bko.gaps` can read every frame."""

from __future__ import annotations

import os
import subprocess
import sys

from tests._corpus import REPO_ROOT


def test_gaps_can_import_every_frame() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    script = (
        "import sys\n"
        "from bko.ontology.frames import (\n"
        "    DIABETES_FIELD_ROWS, DIABETES_FRAME, SYNTHEA_FIELD_ROWS, SYNTHEA_FRAME,\n"
        ")\n"
        "print(sorted(m for m in sys.modules if m == 'bko.evaluate' "
        "or m.startswith('bko.evaluate.')))"
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
