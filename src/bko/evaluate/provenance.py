"""What a `score` or `contradictions` run logs about the code and libraries that produced it."""

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
from bko.evaluate.utility import THRESHOLD


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
"""The distributions whose installed versions a `score` or `contradictions` run logs."""


def log_experiment(name: str, experiment: Experiment, models: Iterable[str], *, frame: str) -> None:
    """Log the run's `experiment_line`, then the commit and libraries it ran with."""
    logger.info("%s", experiment_line(name, experiment, models, frame=frame))
    log_versions()


def log_versions() -> None:
    """Log the `versions` line: the commit, whether its tree is dirty, and `versions()`."""
    head, dirty = commit()
    logger.info(
        "versions commit=%s dirty=%s %s", head, dirty, json.dumps(versions(), sort_keys=True)
    )


def experiment_line(name: str, experiment: Experiment, models: Iterable[str], *, frame: str) -> str:
    """The run's configuration, frame and models, as the `experiment` log line's message.

    `seeds=` is the first and last holdout seed, `<first>-<last>`; `removal=` the removed cell's
    band, sex and label joined by `/`, then `@` and the fraction, or `none`; `cutoff=` follows
    `fill_to=` only when it is not E1's; `replica_rows=` follows `replica=` only when set.
    """
    seeds, removal = experiment.holdout_seeds, experiment.removal
    removed = (
        "none" if removal is None else f"{'/'.join(map(str, removal.cell))}@{removal.fraction}"
    )
    cutoff = "" if experiment.cutoff == THRESHOLD else f"cutoff={experiment.cutoff} "
    capped = "" if experiment.replica_rows is None else f"replica_rows={experiment.replica_rows} "
    return (
        f"experiment run={name} frame={frame} holdouts={len(seeds)} seeds={seeds[0]}-{seeds[-1]} "
        f"removal={removed} training_fraction={experiment.training_fraction} "
        f"decontaminate={'true' if experiment.decontaminate else 'false'} "
        f"arms={','.join(experiment.arms)} replica={','.join(experiment.replica_arms)} {capped}"
        f"threshold={experiment.threshold} fill_to={experiment.fill_to} {cutoff}"
        f"budget={experiment.budget} validity={experiment.validity} "
        f"spend_ceiling={experiment.spend_ceiling:.2f} "
        f"criteria={','.join(experiment.criteria) or 'none'} models={','.join(models)}"
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
