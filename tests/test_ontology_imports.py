"""pronto stays in `bko.ontology.obo`: every other stage reads `bko.ontology.terms` values."""

from __future__ import annotations

import ast
from pathlib import Path

from tests._corpus import REPO_ROOT


def _imports_pronto(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        else:
            continue
        if any(name == "pronto" or name.startswith("pronto.") for name in names):
            return True
    return False


def test_only_obo_imports_pronto() -> None:
    importers = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "src" / "bko").rglob("*.py")
        if _imports_pronto(path)
    }
    assert importers == {Path("src/bko/ontology/obo.py")}
