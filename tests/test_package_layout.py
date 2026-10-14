"""Every planned pipeline stage is an importable `bko` subpackage with a README."""

import importlib

import pytest

from tests._corpus import REPO_ROOT


SUBPACKAGES = (
    "datasets",
    "ontology",
    "bkb",
    "gaps",
    "generate",
    "validate",
    "baselines",
    "evaluate",
    "app",
)
PACKAGE_ROOT = REPO_ROOT / "src" / "bko"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_and_has_a_readme(name: str) -> None:
    importlib.import_module(f"bko.{name}")
    assert (PACKAGE_ROOT / name / "README.md").is_file()


def test_no_subpackage_is_unlisted() -> None:
    found = {path.parent.name for path in PACKAGE_ROOT.glob("*/__init__.py")}
    assert found == set(SUBPACKAGES)


def test_the_package_ships_its_type_marker() -> None:
    assert (PACKAGE_ROOT / "py.typed").is_file()
