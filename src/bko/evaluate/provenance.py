"""What a `score` run logs about the code and libraries that produced it."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from bko.evaluate.experiment import Experiment


logger = logging.getLogger("bko.evaluate")

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
"""The distributions whose installed versions a `score` run logs."""


def log_experiment(name: str, experiment: Experiment, models: Iterable[str]) -> None:
    """Log the run's configuration and models, then the commit and library versions it ran with."""
    logger.info(
        "experiment run=%s holdouts=%d arms=%s replica=%s fill_to=%d budget=%d validity=%s "
        "spend_ceiling=%.2f criteria=%s models=%s",
        name,
        len(experiment.holdout_seeds),
        ",".join(experiment.arms),
        ",".join(experiment.replica_arms),
        experiment.fill_to,
        experiment.budget,
        experiment.validity,
        experiment.spend_ceiling,
        ",".join(experiment.criteria) or "none",
        ",".join(models),
    )
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
