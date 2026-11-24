"""pronto and fastobo stay in `bko.ontology.obo`; every other stage reads `bko.ontology.terms`."""

from __future__ import annotations

import ast
from pathlib import Path

from tests._corpus import REPO_ROOT


_PARSERS = ("pronto", "fastobo")


def _imports_a_parser(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        else:
            continue
        if any(name.split(".")[0] in _PARSERS for name in names):
            return True
    return False


def test_only_obo_imports_an_obo_parser() -> None:
    importers = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "src" / "bko").rglob("*.py")
        if _imports_a_parser(path)
    }
    assert importers == {Path("src/bko/ontology/obo.py")}
