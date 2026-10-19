"""`bko.fusion`'s import boundaries: no SDK, no fuse call, no downstream stage."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from tests._corpus import REPO_ROOT


PACKAGE = REPO_ROOT / "src" / "bko" / "fusion"


def _imported(path: Path) -> set[str]:
    """Every module `path` names in an import statement."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _importers(*modules: str) -> set[Path]:
    """The package files that import one of `modules` or a submodule of one."""
    return {
        path.relative_to(REPO_ROOT)
        for path in PACKAGE.rglob("*.py")
        if any(
            name == module or name.startswith(f"{module}.")
            for name in _imported(path)
            for module in modules
        )
    }


def test_discovery_finds_the_package_imports():
    """The control: the test below asserts a set empty, which a broken walk would also give."""
    assert len(list(PACKAGE.rglob("*.py"))) > 1
    assert Path("src/bko/fusion/__main__.py") in _importers("bko.validate.checks")


def test_fusion_imports_no_sdk_and_no_fuse():
    forbidden = (
        "anthropic",
        "bko.generate.api",
        "bko.bkb.fusion",
        "bko.baselines",
        "bko.evaluate",
        "bko.app",
    )
    assert _importers(*forbidden) == set()


def test_running_fusion_loads_no_sdk():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, bko.fusion.__main__; print('anthropic' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
