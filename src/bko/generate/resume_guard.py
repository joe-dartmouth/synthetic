"""The resume guard: which stopped call logs a resume takes, read from their `run` and `end` lines.

`refuse_unresumable` refuses a replay's log and a rehearsal's log, and takes a ceiling or
projection stop only under the named run's current `Ceiling`, when that ceiling is above the
stopped run's own projection: its logged spend over the holdouts it finished (at least one),
scaled to every holdout, never below the spend itself. `log_ends` reads those two lines alone, so
`launch --resume` refuses before it claims a folder. `bko.generate.replay.Resume` runs the same
guard on the whole log. ADR `research/call-log-replay` records the design.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple

from bko.generate.errors import ReplayError
from bko.generate.log import SCHEMA, SPEND_STOPS


class Ceiling(NamedTuple):
    """The named run's spend rule a resume of a ceiling or projection stop runs under.

    Attributes:
        usd: Its current spend ceiling in USD (`Experiment.spend_ceiling`).
        holdouts: Its holdouts, the count a projection scales to.
    """

    usd: float
    holdouts: int


class Raised(NamedTuple):
    """A ceiling or projection stop a resume takes, read against the raised ceiling.

    Attributes:
        stop: The rule that stopped the run.
        usd: The stopped log's whole spend in USD, its `end` line's `usd`.
        live_usd: What the stopped segment itself paid, its `end` line's `live_usd`.
        holdouts: The holdouts it finished, its `end` line's `holdouts`.
        projected: Its projection in USD: `usd` over `holdouts` (at least 1), scaled to every
            holdout, and never below `usd`.
        ceiling: The raised ceiling in USD, above `projected`.
    """

    stop: str
    usd: float
    live_usd: float
    holdouts: int
    projected: float
    ceiling: float


TAIL: Final = 1 << 16
"""Bytes read from a call log's end to find its `end` line."""


def refuse_unresumable(
    path: Path,
    run: Mapping[str, Any],
    end: Mapping[str, Any] | None,
    ceiling: Ceiling | None = None,
) -> Raised | None:
    """Refuse a stopped log whose `run` or `end` line says it must not be resumed.

    Args:
        path: The stopped log, for the messages.
        run: Its `run` line.
        end: Its `end` line, or None.
        ceiling: The named run's spend rule, or None, which refuses every ceiling or projection
            stop.

    Returns:
        The ceiling or projection stop the resume takes, or None for any other stop.

    Raises:
        ReplayError: If the log replays another or is a rehearsal's, or ended on a ceiling or
            projection stop (`SPEND_STOPS`) with no `ceiling`, no finished holdouts or spend
            recorded, or a projection `ceiling` does not cover.
    """
    source = run.get("replay")
    if source is not None:
        msg = f"{path} replays the log with sha256 {source}; replay that log instead"
        raise ReplayError(msg)
    if run.get("rehearsal"):
        msg = f"{path} is a rehearsal's log, whose replies are canned; a resume sends live"
        raise ReplayError(msg)
    stop = end.get("stop") if end is not None else None
    if end is None or stop not in SPEND_STOPS:
        return None
    if ceiling is None:
        msg = (
            f"{path} was stopped by the {stop} rule, so it resumes only under the named run's "
            "ceiling: raise it past the run's projection and resume with launch --resume"
        )
        raise ReplayError(msg)
    done = end.get("holdouts")
    if isinstance(done, bool) or not isinstance(done, int) or done < 0:
        msg = f"{path} was stopped by the {stop} rule and records no finished holdouts: {done!r}"
        raise ReplayError(msg)
    usd, live = _usd(end.get("usd")), _usd(end.get("live_usd"))
    if usd is None or live is None:
        msg = f"{path} was stopped by the {stop} rule and records no usd and live_usd spend"
        raise ReplayError(msg)
    projected = max(usd, usd / max(done, 1) * ceiling.holdouts)
    if not ceiling.usd > projected:
        msg = (
            f"{path} was stopped by the {stop} rule at a logged spend of {usd:.4f} USD "
            f"(live {live:.4f} USD) over {done} holdouts, which projects to {projected:.4f} USD "
            f"over {ceiling.holdouts}; the named run's ceiling of {ceiling.usd:.2f} USD does not "
            "cover it, so raise the ceiling past the projection"
        )
        raise ReplayError(msg)
    return Raised(str(stop), usd, live, done, projected, ceiling.usd)


def _usd(value: object) -> float | None:
    """`value` as a spend when it is a JSON number, not a boolean; None otherwise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def log_ends(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The `run` line of the call log at `path` and its `end` line, or None when it has none.

    Only the first line and the last `TAIL` bytes are read, so a paid log of hundreds of MB costs
    nothing to check.

    Raises:
        ReplayError: If the first line is not a schema `SCHEMA` `run` line.
        OSError: If the log cannot be read.
    """
    with path.open("rb") as handle:
        first = handle.readline()
        size = handle.seek(0, 2)
        handle.seek(max(0, size - TAIL))
        tail = [line for line in handle.read().splitlines() if line.strip()]
    try:
        run = json.loads(first)
    except ValueError:
        run = None
    if not isinstance(run, dict) or run.get("kind") != "run" or run.get("schema") != SCHEMA:
        msg = f"{path} is not a schema {SCHEMA} call log"
        raise ReplayError(msg)
    try:
        last = json.loads(tail[-1]) if tail else None
    except ValueError:
        last = None
    end = last if isinstance(last, dict) and last.get("kind") == "end" else None
    return run, end
