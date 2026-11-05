"""The stamp on each stamped log line, and the `compare` command's lines over a scores file.

`Stamp` writes each line's local time with its UTC offset. `compare` logs copied test records and
arm means, each for the Runs row that copies it (`docs/EVIDENCE.md`, How to read an entry); no
verdict reads them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Final, TextIO, override

from bko.evaluate.criteria import JUDGED
from bko.evaluate.scores import Measure, Scope, Score
from bko.evaluate.scores import Source as ScoreSource


logger = logging.getLogger("bko.evaluate")

FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"
"""A stamped line: its time, level and logger name, then the message."""


class Stamp(logging.Formatter):
    """Format `%(asctime)s` as `YYYY-MM-DD HH:MM:SS,mmm` local time, then its UTC offset `+HHMM`.

    The stamp holds no space between the milliseconds and the offset, so it stays two tokens.
    """

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """The record's local time, its milliseconds and its UTC offset; `datefmt` is ignored."""
        ct = self.converter(record.created)
        return (
            time.strftime("%Y-%m-%d %H:%M:%S", ct)
            + f",{int(record.msecs):03d}"
            + time.strftime("%z", ct)
        )


def stamped_handler(stream: TextIO) -> logging.Handler:
    """A handler that writes stamped lines to `stream`."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(Stamp(FORMAT))
    return handler


def log_copies(scores: Sequence[Score]) -> None:
    """Log each arm's test records copied, summed over the holdouts, by added and replica rows."""
    totals: dict[str, dict[ScoreSource, float]] = {}
    for s in scores:
        if s.measure is Measure.TEST_COPIES:
            found = totals.setdefault(s.arm, {})
            found[s.source] = found.get(s.source, 0.0) + s.value
    for arm, found in totals.items():
        added, replica = found.get(ScoreSource.ADDED, 0.0), found.get(ScoreSource.REPLICA, 0.0)
        logger.info("copies arm=%s added=%.0f replica=%.0f", arm, added, replica)


def log_means(
    scores: Sequence[Score],
    *,
    source: ScoreSource = ScoreSource.AUGMENTED,
    measures: Sequence[Measure] = JUDGED,
    tag: str = "mean",
) -> None:
    """Log each arm's mean of `measures` over the holdouts, from `source`'s rows, in every scope.

    No verdict reads them. A measure and scope with no value for an arm gets no line.
    """
    for arm in dict.fromkeys(s.arm for s in scores if s.source is source):
        for measure in measures:
            for scope in Scope:
                values = [
                    s.value
                    for s in scores
                    if (s.arm, s.source, s.measure, s.scope) == (arm, source, measure, scope)
                ]
                if not values:
                    continue
                mean = sum(values) / len(values)
                logger.info(
                    "%s arm=%s measure=%s scope=%s value=%.6f holdouts=%d",
                    tag,
                    arm,
                    measure,
                    scope,
                    mean,
                    len(values),
                )
