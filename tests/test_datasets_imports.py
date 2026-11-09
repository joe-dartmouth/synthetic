"""`bko.datasets` imports no `bko.ontology` module, so `bko.ontology` can import it."""

from __future__ import annotations

import os
import subprocess
import sys

from tests._corpus import REPO_ROOT


def test_datasets_imports_no_ontology() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    script = (
        "import importlib, pkgutil, sys, bko.datasets\n"
        "for info in pkgutil.iter_modules(bko.datasets.__path__, 'bko.datasets.'):\n"
        "    importlib.import_module(info.name)\n"
        "print(sorted(m for m in sys.modules if m == 'bko.ontology' "
        "or m.startswith('bko.ontology.')))"
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
