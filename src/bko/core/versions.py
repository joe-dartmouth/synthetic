"""The `versions` line every run logs: the commit, whether its tree is dirty, and the versions.

`log_versions` is the line's one writer, so `bko.gaps` and `bko.evaluate` log the same bytes.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final


logger = logging.getLogger("bko.evaluate")
"""The `versions` line's logger, named `bko.evaluate` as when `bko.evaluate.provenance` held it,
so every run's line keeps its bytes whichever stage logs it."""

VERSIONED: Final = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "imbalanced-learn",
    "sdv",
    "sdmetrics",
    "torch",
    "anthropic",
    "pronto",
)
"""The distributions whose installed versions a run logs."""


def log_versions() -> None:
    """Log the `versions` line: the commit, whether its tree is dirty, and `versions()`."""
    head, dirty = commit()
    logger.info(
        "versions commit=%s dirty=%s %s", head, dirty, json.dumps(versions(), sort_keys=True)
    )


def commit() -> tuple[str, str]:
    """The checkout's HEAD and whether its tree differs from it, or `unknown` for each."""
    here = Path(__file__).parent
    found: list[str] = []
    for command in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"]):
        try:
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                command,
                cwd=here,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except OSError, subprocess.TimeoutExpired:
            return "unknown", "unknown"
        if result.returncode != 0:
            return "unknown", "unknown"
        found.append(result.stdout.strip())
    head, status = found
    return head, "true" if status else "false"


def versions() -> dict[str, str]:
    """The Python, platform and `VERSIONED` distributions' versions, `absent` if not installed."""
    found = {"python": platform.python_version(), "platform": platform.platform()}
    for name in VERSIONED:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "absent"
    return found
