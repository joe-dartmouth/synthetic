"""`bko.gaps`' import boundary: no module of it imports `bko.evaluate`, a downstream stage."""

from __future__ import annotations

import ast
from pathlib import Path

from tests._corpus import REPO_ROOT


PACKAGE = REPO_ROOT / "src" / "bko" / "gaps"


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


def test_discovery_finds_the_package_imports() -> None:
    """The control: the test below asserts a set empty, which a broken walk would also give."""
    assert len(list(PACKAGE.rglob("*.py"))) > 1
    assert _importers("bko.core.versions") == {Path("src/bko/gaps/__main__.py")}


def test_gaps_imports_no_evaluate_module() -> None:
    assert _importers("bko.evaluate") == set()
