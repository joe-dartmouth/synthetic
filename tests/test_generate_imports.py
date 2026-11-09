"""`bko.generate`'s import boundaries: one SDK importer, upstream stages only, no other client."""

from __future__ import annotations

import ast
from pathlib import Path

from tests._corpus import REPO_ROOT


PACKAGE = REPO_ROOT / "src" / "bko" / "generate"


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


def _importers(*modules: str, skip: frozenset[str] = frozenset()) -> set[Path]:
    """The package files, outside `skip`, that import one of `modules` or a submodule of one."""
    return {
        path.relative_to(REPO_ROOT)
        for path in PACKAGE.rglob("*.py")
        if path.name not in skip
        and any(
            name == module or name.startswith(f"{module}.")
            for name in _imported(path)
            for module in modules
        )
    }


def test_discovery_finds_the_package_imports():
    """The control: each test below asserts a set empty, which a broken walk would also give."""
    assert len(list(PACKAGE.rglob("*.py"))) > 1
    assert Path("src/bko/generate/__main__.py") in _importers("bko.gaps")
    assert _importers("bko.gaps.scope") == {Path("src/bko/generate/__main__.py")}


def test_only_api_imports_anthropic():
    assert _importers("anthropic") == {Path("src/bko/generate/api.py")}


def test_generate_imports_no_downstream_stage():
    assert _importers("bko.validate", "bko.fusion", "bko.baselines", "bko.evaluate") == set()


def test_generate_imports_no_release_loader():
    loaders = ("bko.ontology.obo", "bko.ontology.releases", "bko.gaps.scope")
    assert _importers(*loaders, skip=frozenset({"__main__.py"})) == set()


def test_generate_opens_no_other_network_client():
    """`bko.generate.local` alone opens one, `http.client` to the loopback llama.cpp server, and
    parses its address with `urllib.parse`; `bko.generate.reasoning` posts `/tokenize` on the
    connection `local` opened, naming `http.client` for its type and exceptions;
    `bko.generate.weights` names `http.client` for its exceptions and downloads through
    `bko.datasets.fetch.open_https`. No call in `reasoning` opens a connection."""
    clients = ("urllib", "http.client", "httpx2", "httpx", "requests")
    allowed = {
        Path("src/bko/generate/local.py"),
        Path("src/bko/generate/reasoning.py"),
        Path("src/bko/generate/weights.py"),
    }
    assert _importers(*clients) == allowed
    for path in allowed:
        used = {
            name
            for name in _imported(REPO_ROOT / path)
            if any(name == c or name.startswith(f"{c}.") for c in clients)
        }
        assert used <= {"http.client", "urllib.parse", "urllib.parse.urlsplit"}, path
    assert _connections(PACKAGE / "reasoning.py") == []


def _connections(path: Path) -> list[str]:
    """Each call in `path` to a name, or an attribute, ending `Connection`, as written."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id.endswith("Connection"))
            or (isinstance(node.func, ast.Attribute) and node.func.attr.endswith("Connection"))
        )
    ]


def test_the_connection_check_finds_the_local_connections():
    assert _connections(PACKAGE / "local.py").count("http.client.HTTPConnection") >= 1


SOURCE = REPO_ROOT / "src"


def test_only_api_imports_anthropic_anywhere_in_src():
    """Every Messages arm, the CLLM arms included, reaches the SDK only through `bko.generate`."""
    found = {
        path.relative_to(REPO_ROOT)
        for path in SOURCE.rglob("*.py")
        if any(name == "anthropic" or name.startswith("anthropic.") for name in _imported(path))
    }
    assert found == {Path("src/bko/generate/api.py")}


def test_the_src_walk_reads_every_package():
    """The control: the walk above would also give one file if it read `bko.generate` alone."""
    assert Path("src/bko/evaluate/cllm.py") in {
        path.relative_to(REPO_ROOT) for path in SOURCE.rglob("*.py")
    }
    assert "bko.evaluate.cllm" in _imported(SOURCE / "bko" / "evaluate" / "guided.py")
