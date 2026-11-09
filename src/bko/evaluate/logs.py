"""The `compare` command's log lines over a scores file.

`compare` logs copied test records and arm means, each for the Runs row that copies it
(`docs/EVIDENCE.md`, How to read an entry); no verdict reads them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from bko.evaluate.criteria import JUDGED
from bko.evaluate.scores import Measure, Scope, Score
from bko.evaluate.scores import Source as ScoreSource


logger = logging.getLogger("bko.evaluate")


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
