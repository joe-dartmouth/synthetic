"""Run folders: one directory per run of record under `runs/`, each artifact under a role name.

`docs/DEVELOPMENT.md` (Runs) holds the commands that write them and mirrors `ROLES`; ADR
`tooling/run-folders` records the layout.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final


ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "calls.jsonl": "the call log of a run that calls the Messages API",
        "scores.csv": "`python -m bko.evaluate score`'s stdout",
        "scores.log": "its stderr",
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

RUN_ID: Final = re.compile(
    r"(?P<entry>[a-z]\d+)-\d+(?:-[a-z]+(?:-\d+)?)?|app-\d{8}T\d{12}Z-[a-z]+(?:-[a-z]+)*",
    re.ASCII,
)
"""A run folder's name: a register run id such as `e1-3` or `c1-2-closure`, or a demo fill's
`app-<UTC time>-<arm>`. `entry` names the register entry, and is None for a demo fill. Match it
with `fullmatch`; it is ASCII-only, since a match becomes a path."""
