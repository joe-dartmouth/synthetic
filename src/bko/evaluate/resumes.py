"""The checks a resume's stopped log passes before `launch` or a supervisor claims its folder.

`resumable` runs the resume guard (`bko.generate.resume_guard`) on the log's `run` and `end`
lines and compares its settings with the named run's; `superseded` refuses a log a later segment
already resumed; `raised_line` is the `launch.log` line that records a raised ceiling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bko.evaluate.errors import EvaluationError
from bko.generate.errors import ReplayError
from bko.generate.messages import Settings
from bko.generate.resume_guard import Ceiling, Raised, log_ends, refuse_unresumable


def resumable(
    path: Path, settings: Settings, ceiling: Ceiling | None = None, *, run: str
) -> Raised | None:
    """Refuse a stopped log a resume of the named run `run` cannot resume, before any claim.

    `ceiling` is the ceiling raised for this resume, under which a ceiling or projection stop
    resumes; a supervisor's resume passes none, since it never resumes those stops.

    Returns:
        The ceiling or projection stop the resume takes, or None for any other stop.

    Raises:
        EvaluationError: As `bko.generate.resume_guard.refuse_unresumable`, or if the log's settings
            differ from `settings`, the named run's, so its first replayed call would differ.
    """
    try:
        run_line, end = log_ends(path)
        raised = refuse_unresumable(path, run_line, end, ceiling, name=run)
    except ReplayError as exc:
        raise EvaluationError(str(exc)) from exc
    expected = json.loads(json.dumps(asdict(settings)))
    if run_line.get("settings") != expected:
        msg = f"{path} was made with settings {run_line.get('settings')}, not the run's {expected}"
        raise EvaluationError(msg)
    return raised


def superseded(path: Path, root: Path) -> None:
    """Refuse the stopped log at `path` when a run folder under `root` already resumed it.

    A folder's call log whose `run` line names `path`'s sha256 under `resume` continued it, so
    resuming `path` again would send the calls that later segment paid for; the last segment's
    log is the one to resume.

    Raises:
        EvaluationError: If such a folder exists.
        OSError: If `path` cannot be read.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    for log in sorted(root.glob("*/calls.jsonl")):
        try:
            with log.open("rb") as handle:
                first = json.loads(handle.readline())
        except OSError, ValueError:
            continue
        prior = first.get("resume") if isinstance(first, dict) else None
        if isinstance(prior, dict) and prior.get("sha256") == digest:
            msg = (
                f"{log} already resumed {path} (sha256 {digest}); resume the last segment's log, "
                "not an earlier one, so no paid call is sent twice"
            )
            raise EvaluationError(msg)


def raised_line(raised: Raised) -> str:
    """The `launch.log` line that records the raised ceiling a resume accepted, and why."""
    return (
        f"ceiling={raised.ceiling:.4f} stop={raised.stop} usd={raised.usd:.4f} "
        f"live_usd={raised.live_usd:.4f} holdouts={raised.holdouts} "
        f"projected_usd={raised.projected:.4f}"
    )
