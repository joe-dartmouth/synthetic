"""Every planned pipeline stage is an importable `bko` subpackage with a README.

The stage table in ADR `tooling/src-layout`, `SUBPACKAGES`, the directories under `src/bko/`, each
README's declared extra and `pyproject.toml`'s extras are held equal here, so none can drift alone.
"""

from __future__ import annotations

import importlib
import re
import tomllib

import pytest

from tests._corpus import REPO_ROOT


SUBPACKAGES = (
    "datasets",
    "ontology",
    "bkb",
    "gaps",
    "generate",
    "validate",
    "fusion",
    "baselines",
    "evaluate",
    "app",
)
PACKAGE_ROOT = REPO_ROOT / "src" / "bko"
LAYOUT_ADR = REPO_ROOT / "docs" / "adr" / "tooling" / "src-layout.md"

README_FIELDS = ("- **Dependencies:**", "- **Owner:**", "- **Status:**")
"""The lines every stage README carries after its role paragraph."""

_ADR_ROW = re.compile(r"^\|\s*`([a-z]+)`\s*\|[^|]*\|\s*(?:`([a-z]+)`|none)\s*\|\s*$", re.MULTILINE)
"""One row of the ADR's stage table: the subpackage, its stage, and its extra or `none`."""

_README_EXTRA = re.compile(r"^- \*\*Dependencies:\*\* (?:The `([a-z]+)` extra|No optional extra)")
"""A README's dependency line: the extra it will import, or none."""


def adr_extras() -> dict[str, str | None]:
    """Subpackage -> extra, read from the ADR's stage table."""
    return {name: extra or None for name, extra in _ADR_ROW.findall(LAYOUT_ADR.read_text())}


def readme_extra(name: str) -> str | None:
    """The extra one stage README declares, failing when it declares none in the known form."""
    text = (PACKAGE_ROOT / name / "README.md").read_text(encoding="utf-8")
    match = next(
        (m for line in text.splitlines() if (m := _README_EXTRA.match(line)) is not None), None
    )
    assert match is not None, f"src/bko/{name}/README.md has no recognizable Dependencies line"
    return match.group(1)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_and_has_a_readme(name: str) -> None:
    importlib.import_module(f"bko.{name}")
    readme = PACKAGE_ROOT / name / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    missing = [field for field in README_FIELDS if field not in text]
    assert not missing, f"src/bko/{name}/README.md lacks {missing}"


def test_no_subpackage_is_unlisted() -> None:
    """Every directory under `src/bko/` is a listed subpackage, `__init__.py` or not."""
    found = {
        path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert found == set(SUBPACKAGES)
    missing_init = sorted(
        name for name in found if not (PACKAGE_ROOT / name / "__init__.py").is_file()
    )
    assert not missing_init, f"stage directories without `__init__.py`: {missing_init}"


def test_the_layout_adr_lists_exactly_the_subpackages() -> None:
    """ADR `tooling/src-layout`'s table names the same stages as `SUBPACKAGES`, in order."""
    assert list(adr_extras()) == list(SUBPACKAGES)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_each_readme_declares_the_extra_the_adr_names(name: str) -> None:
    """A stage's extra has two statements, the ADR row and the README; they must agree."""
    assert readme_extra(name) == adr_extras()[name]


def test_every_named_extra_exists_in_pyproject() -> None:
    declared = set(
        tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    named = {extra for extra in adr_extras().values() if extra is not None}
    assert named <= declared, f"extras named by the layout but not declared: {named - declared}"


def test_the_package_ships_its_type_marker() -> None:
    assert (PACKAGE_ROOT / "py.typed").is_file()
