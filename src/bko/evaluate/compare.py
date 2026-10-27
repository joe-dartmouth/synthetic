"""`compare`: judge a scores file against an experiment's criteria and log what it holds.

With `--beside`, it also logs the judged arm read beside an arm of another scores file
(`bko.evaluate.beside`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bko.evaluate.beside import Pairing, Skipped, check, pair, pairable
from bko.evaluate.criteria import JUDGED, held_out_ratio_of, judge, series_of, table
from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.experiment import BKO_ARMS, Experiment
from bko.evaluate.fidelity import MEASURES as FIDELITY
from bko.evaluate.logs import log_copies, log_means
from bko.evaluate.scores import Score, Source, parse
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
    reading beside the other file's arm is checked before the table, so a reading it refuses leaves
    no table, and logged last; it never changes the table.

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
    candidate = experiment.candidate if candidate is None else candidate
    logger.info(
        "compare candidate=%s dry_run=%s scores_sha256=%s",
        candidate,
        candidate != experiment.candidate,
        hashlib.sha256(data).hexdigest(),
    )
    rows = parse(text)
    contrasts = judge(rows, experiment, candidate=candidate)
    reading = (
        None
        if beside is None
        else _prepare(beside, rows, arm=candidate, against=beside_arm or candidate)
    )
    for criterion in dict.fromkeys(contrast.criterion for contrast in contrasts):
        outcome = next(c.verdict for c in contrasts if c.criterion == criterion)
        logger.info("criterion=%s verdict=%s", criterion, outcome)
    out.write(table(contrasts))
    _log_validity(rows, experiment, candidate)
    log_copies(rows)
    log_means(rows)
    log_means(rows, source=Source.REPLICA, measures=(*JUDGED, *FIDELITY), tag="replica")
    if reading is not None:
        _log_beside(reading, rows, experiment)


@dataclass(frozen=True, slots=True)
class _Reading:
    """The other scores file of `--beside`, and which of its holdouts pair."""

    scores: list[Score]
    digest: str
    arm: str
    against: str
    pairing: Pairing


def _text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"the scores are not UTF-8: {exc}"
        raise ScoresFormatError(msg) from exc


def _prepare(path: Path, ours: Sequence[Score], *, arm: str, against: str) -> _Reading:
    """Read the other file and pair its holdouts; on a refusal, log the counts first.

    Raises:
        EvaluationError: As `bko.evaluate.beside.check`.
        ScoresFormatError: If the file is not UTF-8 or breaks the scores' format.
    """
    data = path.read_bytes()
    theirs = parse(_text(data))
    found = _Reading(theirs, hashlib.sha256(data).hexdigest(), arm, against, pairable(ours, theirs))
    try:
        check(ours, theirs, arm=arm, against=against, holdouts=found.pairing.holdouts)
    except EvaluationError:
        _log_counts(found)
        raise
    return found


def _log_counts(reading: _Reading) -> None:
    pairing = reading.pairing
    logger.info(
        "beside scores_sha256=%s shared=%d paired=%d excluded=%d copied=%d mismatched=%d "
        "arm=%s against=%s",
        reading.digest,
        pairing.shared,
        len(pairing.holdouts),
        pairing.excluded,
        pairing.copied,
        pairing.mismatched,
        reading.arm,
        reading.against,
    )


def _log_beside(reading: _Reading, ours: Sequence[Score], experiment: Experiment) -> None:
    """Log the pairing's counts, then each series read beside the other file's arm, or skipped."""
    _log_counts(reading)
    arm, against = reading.arm, reading.against
    found = pair(
        ours,
        reading.scores,
        arm=arm,
        against=against,
        holdouts=reading.pairing.holdouts,
        ratio=held_out_ratio_of(series_of(ours)),
        confidence=experiment.confidence,
    )
    for series in found:
        if isinstance(series, Skipped):
            logger.info(
                "beside skip arm=%s against=%s source=%s measure=%s scope=%s reason=%s defined=%d",
                arm,
                against,
                series.source,
                series.measure,
                series.scope,
                series.reason,
                series.defined,
            )
            continue
        logger.info(
            "beside arm=%s against=%s source=%s measure=%s scope=%s mean=%.6f low=%.6f high=%.6f "
            "holdouts=%d differing=%d undefined=%d",
            arm,
            against,
            series.source,
            series.measure,
            series.scope,
            series.interval.mean,
            series.interval.low,
            series.interval.high,
            series.interval.holdouts,
            series.differing,
            series.undefined,
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
