"""Where run folders live and what their names look like.

`bko.app.runs` holds the folders' roles and reads them back (`docs/DEVELOPMENT.md`, Runs; ADR
`tooling/run-folders`); the evaluation's `launch` writes a folder, so the name and the root live
below both.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

from bko.datasets.fetch import default_data_dir


RUN_ID: Final = re.compile(
    r"(?P<entry>[a-z]\d+)-\d+(?:-[a-z]+(?:-\d+)?)?|app-\d{8}T\d{12}Z-[a-z]+(?:-[a-z]+)*",
    re.ASCII,
)
"""A run folder's name: a register run id such as `e1-3` or `c1-2-closure`, or a demo fill's
`app-<UTC time>-<arm>`. `entry` names the register entry, and is None for a demo fill. Match it
with `fullmatch`; it is ASCII-only, since a match becomes a path."""


def runs_root() -> Path:
    """`runs/` at the repository root.

    Raises:
        UnavailableError: As `default_data_dir`, outside a checkout.
    """
    return default_data_dir().parent / "runs"


def runs_dir_argument(text: str) -> Path:
    """`--runs-dir`'s argparse type: the run folders' root, resolved.

    Raises:
        argparse.ArgumentTypeError: If `text` is empty, which would name the working directory.
    """
    if not text:
        msg = "the runs root must not be empty"
        raise argparse.ArgumentTypeError(msg)
    return Path(text).resolve()
