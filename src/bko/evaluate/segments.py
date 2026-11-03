"""What a supervised run's finished segment's call log says (`bko.evaluate.supervise`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from bko.generate.errors import ReplayError
from bko.generate.log import SPEND_STOPS
from bko.generate.resume_guard import log_ends


logger = logging.getLogger("bko.evaluate.supervise")


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a finished segment's call log says.

    Attributes:
        status: The `end` line's status; `none` for a log with no `end` line, `absent` for no
            log, `unreadable` for a log whose first line is not a call log's.
        stop: The stop rule the `end` line names, or None.
        retryable: Whether the API caused the stop, or the log has no `end` line and a signal
            killed the child: the segment is resumed while resumes remain.
        rate_limited: Whether any retry in the segment met a 429.
        usd: The `end` line's whole spend in USD, or None.
        holdouts: The `end` line's finished holdouts, or None.
    """

    status: str
    stop: str | None
    retryable: bool
    rate_limited: bool
    usd: float | None = None
    holdouts: int | None = None


def outcome(log: Path, code: int) -> Outcome:
    """Read a finished segment's call log at `log`; its child process, which exited `code`, is gone.

    A log with no `end` line is retryable only when `code` is negative, a signal's: `score` turns
    `SIGTERM`, `SIGINT` and `SIGHUP` into an `interrupted` end line, so a signal that leaves none
    is a kill, and a child that exited on its own with none crashed.
    """
    if not log.is_file():
        return Outcome("absent", None, retryable=False, rate_limited=False)
    try:
        _, end = log_ends(log)
    except (ReplayError, OSError) as exc:
        logger.warning("supervise log=%s unreadable error=%s: %s", log, type(exc).__name__, exc)
        return Outcome("unreadable", None, retryable=False, rate_limited=False)
    if end is None:
        return Outcome("none", None, retryable=code < 0, rate_limited=False)
    stop = end.get("stop")
    status = str(end.get("status"))
    retryable = status == "failed" and end.get("retryable") is True and stop not in SPEND_STOPS
    usd, done = end.get("usd"), end.get("holdouts")
    return Outcome(
        status,
        stop if isinstance(stop, str) else None,
        retryable=retryable,
        rate_limited=end.get("rate_limited") is True,
        usd=float(usd) if isinstance(usd, int | float) and not isinstance(usd, bool) else None,
        holdouts=done if isinstance(done, int) and not isinstance(done, bool) else None,
    )
