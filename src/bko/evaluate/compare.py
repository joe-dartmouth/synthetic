"""`compare`: judge a scores file against an experiment's criteria and log what it holds."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from bko.evaluate.criteria import JUDGED, judge, table
from bko.evaluate.errors import ScoresFormatError
from bko.evaluate.experiment import BKO_ARMS, Experiment
from bko.evaluate.fidelity import MEASURES as FIDELITY
from bko.evaluate.logs import log_copies, log_means
from bko.evaluate.scores import Score, parse
from bko.evaluate.scores import Source as ScoreSource
from bko.evaluate.validity import Validity, reported, validity, verdict


logger = logging.getLogger("bko.evaluate")


def compare(experiment: Experiment, candidate: str | None, scores: Path, out: TextIO) -> None:
    """Write the criteria's table to `out`, then log validity, copies, means and replica means.

    The table goes out first, so scores whose validity cannot be read keep it.

    Args:
        experiment: The configuration the scores are judged against.
        candidate: The row-adding arm judged; the experiment's own when None.
        scores: The CSV that `score` wrote.
        out: Where the table goes.
    """
    data = scores.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"the scores are not UTF-8: {exc}"
        raise ScoresFormatError(msg) from exc
    candidate = experiment.candidate if candidate is None else candidate
    logger.info(
        "compare candidate=%s dry_run=%s scores_sha256=%s",
        candidate,
        candidate != experiment.candidate,
        hashlib.sha256(data).hexdigest(),
    )
    rows = parse(text)
    contrasts = judge(rows, experiment, candidate=candidate)
    for criterion in dict.fromkeys(contrast.criterion for contrast in contrasts):
        outcome = next(c.verdict for c in contrasts if c.criterion == criterion)
        logger.info("criterion=%s verdict=%s", criterion, outcome)
    out.write(table(contrasts))
    _log_validity(rows, experiment, candidate)
    log_copies(rows)
    log_means(rows)
    log_means(rows, source=ScoreSource.REPLICA, measures=(*JUDGED, *FIDELITY), tag="replica")


def _log_validity(scores: Sequence[Score], experiment: Experiment, candidate: str) -> None:
    """Judge the candidate's validity if it is a BKO arm, and log every other arm's counts.

    Each arm's line is followed by its counts and rate per fill, with no verdict.
    """
    arms = reported(scores)
    if candidate in BKO_ARMS or candidate in arms:
        found = validity(scores, experiment, candidate)
        if "validity" in experiment.criteria:
            logger.info(
                "criterion=validity verdict=%s %s", verdict(found, experiment), found.line()
            )
        else:
            logger.info("validity %s", found.line())
        _log_fills(found)
    for arm in arms:
        if arm != candidate:
            found = validity(scores, experiment, arm)
            logger.info("validity %s", found.line())
            _log_fills(found)


def _log_fills(found: Validity) -> None:
    for line in found.fill_lines():
        logger.info("candidates %s", line)
