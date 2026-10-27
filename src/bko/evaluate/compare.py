"""`compare`: judge a scores file against an experiment's criteria and log what it holds.

With `--beside`, it also logs the judged arm read beside an arm of another scores file
(`bko.evaluate.beside`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from bko.evaluate.beside import pair, pairable
from bko.evaluate.criteria import JUDGED, held_out_ratio_of, judge, series_of, table
from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.experiment import BKO_ARMS, Experiment
from bko.evaluate.fidelity import MEASURES as FIDELITY
from bko.evaluate.logs import log_copies, log_means
from bko.evaluate.scores import Score, parse
from bko.evaluate.scores import Source as ScoreSource
from bko.evaluate.validity import Validity, reported, validity, verdict


logger = logging.getLogger("bko.evaluate")


def compare(
    experiment: Experiment,
    candidate: str | None,
    scores: Path,
    out: TextIO,
    *,
    beside: Path | None = None,
    beside_arm: str | None = None,
) -> None:
    """Write the criteria's table to `out`, then log validity, copies, means and replica means.

    The table goes out first, so scores whose validity cannot be read keep it. With `beside`, the
    reading beside the other file's arm is logged last; it never changes the table.

    Args:
        experiment: The configuration the scores are judged against.
        candidate: The row-adding arm judged; the experiment's own when None.
        scores: The CSV that `score` wrote.
        out: Where the table goes.
        beside: Another scores file to read the judged arm beside.
        beside_arm: The arm of `beside` read; the judged arm when None.

    Raises:
        EvaluationError: If `beside_arm` is given without `beside`, or as `judge` and `pair`.
        ScoresFormatError: If either scores file is not UTF-8 or breaks the scores' format.
    """
    if beside_arm is not None and beside is None:
        msg = "--beside-arm names an arm of the --beside scores, so it needs --beside"
        raise EvaluationError(msg)
    data = scores.read_bytes()
    text = _text(data)
    other = None if beside is None else beside.read_bytes()
    theirs = None if other is None else parse(_text(other))
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
    if other is not None and theirs is not None:
        against = candidate if beside_arm is None else beside_arm
        _log_beside(rows, theirs, other, experiment, arm=candidate, against=against)


def _text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"the scores are not UTF-8: {exc}"
        raise ScoresFormatError(msg) from exc


def _log_beside(
    ours: Sequence[Score],
    theirs: Sequence[Score],
    data: bytes,
    experiment: Experiment,
    *,
    arm: str,
    against: str,
) -> None:
    """Log the pairing's counts, then each series read beside the other file's arm."""
    holdouts, excluded = pairable(ours, theirs)
    logger.info(
        "beside scores_sha256=%s shared=%d paired=%d excluded=%d arm=%s against=%s",
        hashlib.sha256(data).hexdigest(),
        len(holdouts) + excluded,
        len(holdouts),
        excluded,
        arm,
        against,
    )
    ratio = held_out_ratio_of(series_of(ours))
    readings = pair(
        ours,
        theirs,
        arm=arm,
        against=against,
        holdouts=holdouts,
        ratio=ratio,
        confidence=experiment.confidence,
    )
    for found in readings:
        logger.info(
            "beside arm=%s against=%s source=%s measure=%s scope=%s mean=%.6f low=%.6f high=%.6f "
            "holdouts=%d differing=%d undefined=%d",
            arm,
            against,
            found.source,
            found.measure,
            found.scope,
            found.interval.mean,
            found.interval.low,
            found.interval.high,
            found.interval.holdouts,
            found.differing,
            found.undefined,
        )


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
