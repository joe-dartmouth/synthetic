"""The resume guard: which stopped call logs a resume takes, read from their `run` and `end` lines.

`refuse_unresumable` refuses a replay's log, a rehearsal's log and a log of another named run, and
takes a ceiling or projection stop only under a `Ceiling` the operator raised for that resume alone
(`--ceiling`), when that ceiling is above the stopped run's own projection: its logged spend over
the holdouts it finished (at least one), scaled to every holdout, never below the spend itself. Any
other stop refuses a raised ceiling and runs under the ceiling its log already ran under
(`ran_under`). `log_ends` reads those two lines alone, so `launch --resume` refuses before it claims
a folder. `bko.generate.replay.Resume` runs the same guard on the whole log. ADR
`research/call-log-replay` records the design.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple

from bko.generate.errors import ReplayError
from bko.generate.log import SCHEMA, SPEND_STOPS


logger = logging.getLogger(__name__)


class Ceiling(NamedTuple):
    """The raised spend ceiling a resume of a ceiling or projection stop runs under.

    Attributes:
        usd: The ceiling in USD the operator raised for this resume alone (`--ceiling`).
        holdouts: The named run's holdouts, the count a projection scales to.
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
    *,
    name: str | None = None,
) -> Raised | None:
    """Refuse a stopped log whose `run` or `end` line says it must not be resumed.

    Args:
        path: The stopped log, for the messages.
        run: Its `run` line.
        end: Its `end` line, or None.
        ceiling: The ceiling raised for this resume, or None, which refuses every ceiling or
            projection stop.
        name: The named run the resume runs, or None, which checks no run name.

    Returns:
        The ceiling or projection stop the resume takes, or None for any other stop.

    Raises:
        ReplayError: If the log replays another, is a rehearsal's, or names a run other than
            `name`; if it ended on a ceiling or projection stop (`SPEND_STOPS`) with no `ceiling`,
            no run name to check against `name`, no finished holdouts or spend recorded as finite
            numbers of at least 0, or a projection `ceiling` does not cover; or if `ceiling` is
            given for any other stop.
    """
    source = run.get("replay")
    if source is not None:
        msg = f"{path} replays the log with sha256 {source}; replay that log instead"
        raise ReplayError(msg)
    if run.get("rehearsal"):
        msg = f"{path} is a rehearsal's log, whose replies are canned; a resume sends live"
        raise ReplayError(msg)
    logged = run.get("run")
    if name is not None and logged is not None and logged != name:
        msg = f"{path} holds the calls of the run {logged!r}, not {name!r}; resume it as {logged}"
        raise ReplayError(msg)
    stop = end.get("stop") if end is not None else None
    if end is None or stop not in SPEND_STOPS:
        if ceiling is not None:
            msg = (
                f"{path} was not stopped by the ceiling or projection rule (stop={stop}), so it "
                "takes no raised ceiling; resume it without --ceiling"
            )
            raise ReplayError(msg)
        return None
    if ceiling is None:
        msg = (
            f"{path} was stopped by the {stop} rule, so it resumes only under a ceiling raised "
            "past its projection: resume it with launch --resume and --ceiling"
        )
        raise ReplayError(msg)
    if name is not None and logged is None:
        msg = f"{path} names no run, so it cannot be checked against {name!r}; it does not resume"
        raise ReplayError(msg)
    done = end.get("holdouts")
    if isinstance(done, bool) or not isinstance(done, int) or done < 0:
        msg = f"{path} was stopped by the {stop} rule and records no finished holdouts: {done!r}"
        raise ReplayError(msg)
    usd, live = _usd(end.get("usd")), _usd(end.get("live_usd"))
    if usd is None or live is None:
        msg = (
            f"{path} was stopped by the {stop} rule and records no usd and live_usd spend of at "
            f"least 0: {end.get('usd')!r} and {end.get('live_usd')!r}"
        )
        raise ReplayError(msg)
    projected = max(usd, usd / max(done, 1) * ceiling.holdouts)
    if not ceiling.usd > projected:
        partial = (
            " (no holdout finished, so the spend of the holdout in progress is scaled as one)"
            if done == 0
            else ""
        )
        msg = (
            f"{path} was stopped by the {stop} rule at a logged spend of {usd:.4f} USD "
            f"(live {live:.4f} USD) over {done} holdouts, which projects to {projected:.4f} USD "
            f"over {ceiling.holdouts}{partial}; the raised ceiling of {ceiling.usd:.4f} USD does "
            "not cover it, so raise it past the projection"
        )
        raise ReplayError(msg)
    return Raised(str(stop), usd, live, done, projected, ceiling.usd)


def ran_under(path: Path, run: Mapping[str, Any]) -> float | None:
    """The raised ceiling in USD a resumed log's `run` line records under `resume`, or None.

    A resume of a stop other than a ceiling or projection stop runs under it again, so a raised
    ceiling holds across a supervisor's resumes without being raised anew.

    Raises:
        ReplayError: If the recorded ceiling is not a finite number above 0.
    """
    prior = run.get("resume")
    if not isinstance(prior, Mapping) or "ceiling" not in prior:
        return None
    usd = _usd(prior["ceiling"])
    if usd is None or usd == 0:
        msg = f"the run line of {path} records the raised ceiling {prior['ceiling']!r}, not a USD"
        raise ReplayError(msg)
    return usd


def _usd(value: object) -> float | None:
    """`value` as a spend when it is a finite number of at least 0, not a boolean, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    spend = float(value)
    return spend if math.isfinite(spend) and spend >= 0 else None


def log_ends(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The `run` line of the call log at `path` and its `end` line, or None when it has none.

    Only the first line and the last `TAIL` bytes are read, so a paid log of hundreds of MB costs
    nothing to check. A last line with no newline, which a killed process can leave, is no `end`
    line, and is logged as a WARNING.

    Raises:
        ReplayError: If the first line is not a schema `SCHEMA` `run` line.
        OSError: If the log cannot be read.
    """
    with path.open("rb") as handle:
        first = handle.readline()
        size = handle.seek(0, 2)
        handle.seek(max(0, size - TAIL))
        data = handle.read()
    try:
        run = json.loads(first)
    except ValueError:
        run = None
    if not isinstance(run, dict) or run.get("kind") != "run" or run.get("schema") != SCHEMA:
        msg = f"{path} is not a schema {SCHEMA} call log"
        raise ReplayError(msg)
    if not data.endswith(b"\n"):
        cut = len(data) - data.rfind(b"\n") - 1
        logger.warning("call log %s ends in a cut line of %d bytes, not an end line", path, cut)
        return run, None
    tail = [line for line in data.splitlines() if line.strip()]
    try:
        last = json.loads(tail[-1]) if tail else None
    except ValueError:
        last = None
    end = last if isinstance(last, dict) and last.get("kind") == "end" else None
    return run, end
