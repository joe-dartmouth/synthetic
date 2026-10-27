"""Run folders: one directory per run of record under `runs/`, each artifact under a role name.

`docs/DEVELOPMENT.md` (Runs) holds the commands that write them and mirrors `ROLES`; ADR
`tooling/run-folders` records the layout.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from bko.datasets.run_folders import RUN_ID


ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "calls.jsonl": "the call log of a run that calls the Messages API",
        "scores.csv": "`python -m bko.evaluate score`'s stdout",
        "scores.log": "its stderr",
        "holdout-candidates.jsonl": "every candidate its BKO arms generated, per holdout",
        "launch.log": "`python -m bko.evaluate launch`'s record of the power reading and command",
        "criteria.csv": "`python -m bko.evaluate compare`'s stdout",
        "criteria.log": "its stderr",
        "candidates.jsonl": "`python -m bko.generate`'s stdout",
        "candidates.log": "its stderr",
        "closure.csv": "`python -m bko.fusion`'s stdout",
        "closure.log": "its stderr",
        "gaps-uci.csv": "`python -m bko.gaps uci`'s stdout",
        "gaps-uci.log": "its stderr",
        "gaps-synthea.csv": "`python -m bko.gaps synthea`'s stdout",
        "gaps-synthea.log": "its stderr",
        "reproduction.csv": "`python -m bko.ontology reproduce`'s stdout",
        "reproduction.log": "its stderr",
        "training.csv": "the augmented training set of an app fill",
        "pytest": "a `live` test's `--basetemp` directory",
    }
)
"""Each artifact's name in a run folder, in reading order, and what writes it."""

TAIL: Final = 1 << 16
"""Bytes read from a call log's end to find its `end` line; paid logs run to hundreds of MB."""


@dataclass(frozen=True, slots=True)
class Run:
    """One run folder.

    Attributes:
        name: The folder's name, a run id.
        entry: The register entry it belongs to (`e1`, `c1`, ...), or `app` for a demo fill.
        folder: The folder.
        roles: Each role the folder holds, in `ROLES` order.
    """

    name: str
    entry: str
    folder: Path
    roles: Mapping[str, Path]


def discover(root: Path) -> tuple[Run, ...]:
    """The run folders under `root`, by name; none when `root` does not exist.

    A directory whose name is not a run id, a file whose name is not a role, and a symbolic link
    are skipped. Names sort by their numbers, so `e1-10` follows `e1-9`.
    """
    if not root.is_dir():
        return ()
    found = []
    for folder in sorted(root.iterdir(), key=_natural):
        match = RUN_ID.fullmatch(folder.name)
        if match is None or folder.is_symlink() or not folder.is_dir():
            continue
        roles = {role: folder / role for role in ROLES if _holds(folder / role, role)}
        found.append(Run(folder.name, match["entry"] or "app", folder, MappingProxyType(roles)))
    return tuple(found)


def call_totals(path: Path) -> Mapping[str, object] | None:
    """The `end` line of the call log at `path`; None when it has none, as a run still going has."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - TAIL))
        lines = [line for line in handle.read().splitlines() if line.strip()]
    try:
        last = json.loads(lines[-1]) if lines else None
    except ValueError:
        return None
    return last if isinstance(last, dict) and last.get("kind") == "end" else None


def _holds(path: Path, role: str) -> bool:
    """Whether `path` is the role's kind of entry: a directory for `pytest`, else a regular file."""
    if path.is_symlink():
        return False
    return path.is_dir() if role == "pytest" else path.is_file()


def _natural(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", path.name)
    )
