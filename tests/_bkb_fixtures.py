"""The committed pybkb fixtures for the `bko.bkb` tests, read once per process.

The JSON under `tests/pybkb/` was written by `tests/pybkb/export.py` from pybkb at the commit its
manifest names (`docs/DEVELOPMENT.md`, BKB fixtures).
"""

from __future__ import annotations

import functools
import json
from typing import Any

from tests._corpus import REPO_ROOT


PYBKB_DIR = REPO_ROOT / "tests" / "pybkb"
"""Where the exporter's output is committed."""

NOT_DOCUMENTS = frozenset({"manifest", "queries", "run"})


@functools.cache
def fixture_text(name: str) -> str:
    return (PYBKB_DIR / f"{name}.json").read_text(encoding="utf-8")


def document_names() -> list[str]:
    return sorted(p.stem for p in PYBKB_DIR.glob("*.json") if p.stem not in NOT_DOCUMENTS)


def manifest() -> dict[str, Any]:
    return json.loads(fixture_text("manifest"))
