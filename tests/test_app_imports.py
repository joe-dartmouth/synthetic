"""`bko.app`'s import boundaries: one streamlit importer, no SDK or command module, no shadowing."""

from __future__ import annotations

import ast
import importlib.metadata
import sys
from pathlib import Path

from tests._corpus import REPO_ROOT


SRC = REPO_ROOT / "src" / "bko"
PACKAGE = SRC / "app"


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


def _imports(path: Path, module: str) -> bool:
    return any(name == module or name.startswith(f"{module}.") for name in _imported(path))


def _app_importers(module: str) -> set[str]:
    """The names of the app's files that import `module` or a submodule of it."""
    return {path.name for path in PACKAGE.rglob("*.py") if _imports(path, module)}


def test_discovery_finds_the_app_imports():
    """The control: the tests below assert sets empty, which a broken walk would also give."""
    assert len(list(PACKAGE.rglob("*.py"))) > 1
    assert "context.py" in _app_importers("bko.evaluate.run")


def test_only_the_page_imports_streamlit():
    assert _app_importers("streamlit") == {"page.py"}


def test_the_app_imports_no_sdk_and_no_command_module():
    assert _app_importers("anthropic") == set()
    commands = {
        path.name
        for path in PACKAGE.rglob("*.py")
        if any(name.endswith(".__main__") or ".__main__." in name for name in _imported(path))
    }
    assert commands == set()


def test_no_stage_imports_the_app():
    importers = {
        path.relative_to(REPO_ROOT)
        for path in SRC.rglob("*.py")
        if PACKAGE not in path.parents and _imports(path, "bko.app")
    }
    assert importers == set()


def test_no_app_module_shadows_a_top_level_name():
    """`streamlit run` puts the script's directory first on `sys.path`, so a module there named
    like a top-level module or distribution would replace it for every import."""
    taken = set(sys.stdlib_module_names) | set(importlib.metadata.packages_distributions())
    stems = {path.stem for path in PACKAGE.glob("*.py")}
    assert stems & taken == set()


def test_the_page_renders_no_html():
    """Untrusted text reaches the page only as code blocks and frames, never as markup."""
    offenders = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "unsafe_allow_html":
                offenders.add((path.name, node.arg))
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "st"
                and node.attr in {"html", "components"}
            ):
                offenders.add((path.name, node.attr))
    assert offenders == set()
