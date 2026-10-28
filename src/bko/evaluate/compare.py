"""`compare`: judge a scores file against an experiment's criteria and log what it holds.

With `--beside`, it also logs the judged arm read beside an arm of another scores file
(`bko.evaluate.beside`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from bko.evaluate.beside import Pairing, Skipped, check, pair, pairable
from bko.evaluate.criteria import (
    BOOSTED,
    JUDGED,
    Absent,
    Reported,
    held_out_ratio_of,
    judge,
    reported_contrasts,
    series_of,
    table,
)
from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.experiment import CRITERION_NAMES, MESSAGES_ARMS, Experiment
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
    run: str,
    level: str,
    beside: Path | None = None,
    beside_arm: str | None = None,
) -> None:
    """Write the criteria's table to `out`, then log validity, copies, means and replica means.

    The boosting model's contrasts are read after the table and logged after the means when the
    scores hold its rows; they never change the table, and a pair of arms either of which lacks
    them is logged as skipped. An experiment that judges no criterion, such as `g2`, has a table of
    its header alone; every interval of `CRITERION_NAMES` is then logged after the boosting
    model's contrasts as an `unjudged` line, with no verdict.

    The table goes out first, so scores whose validity cannot be read keep it. With `beside`, the
    reading beside the other file's arm is checked before the table, so a reading it refuses leaves
    no table, and logged last; it never changes the table.

    Args:
        experiment: The configuration the scores are judged against.
        candidate: The row-adding arm judged; the experiment's own when None.
        scores: The CSV that `score` wrote.
        out: Where the table goes.
        run: The name of the run the scores belong to; the `beside` lines name it.
        level: The guidance level of that run's settings; the `beside` lines name it.
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
        else _prepare(
            beside, rows, arm=candidate, against=beside_arm or candidate, run=run, level=level
        )
    )
    for criterion in dict.fromkeys(contrast.criterion for contrast in contrasts):
        outcome = next(c.verdict for c in contrasts if c.criterion == criterion)
        logger.info("criterion=%s verdict=%s", criterion, outcome)
    out.write(table(contrasts))
    _log_validity(rows, experiment, candidate)
    log_copies(rows)
    log_means(rows, measures=(*JUDGED, *BOOSTED))
    _log_boosted(reported_contrasts(rows, experiment, candidate=candidate))
    if not experiment.criteria:
        _log_unjudged(rows, experiment, candidate)
    log_means(rows, source=Source.REPLICA, measures=(*JUDGED, *FIDELITY), tag="replica")
    if reading is not None:
        _log_beside(reading, rows, experiment, run=run, level=level)


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


def _prepare(
    path: Path, ours: Sequence[Score], *, arm: str, against: str, run: str, level: str
) -> _Reading:
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
        _log_counts(found, run, level)
        raise
    return found


def _log_counts(reading: _Reading, run: str, level: str) -> None:
    pairing = reading.pairing
    logger.info(
        "beside scores_sha256=%s shared=%d paired=%d excluded=%d copied=%d mismatched=%d "
        "arm=%s against=%s run=%s level=%s",
        reading.digest,
        pairing.shared,
        len(pairing.holdouts),
        pairing.excluded,
        pairing.copied,
        pairing.mismatched,
        reading.arm,
        reading.against,
        run,
        level,
    )


def _log_beside(
    reading: _Reading, ours: Sequence[Score], experiment: Experiment, *, run: str, level: str
) -> None:
    """Log the pairing's counts, then each series read beside the other file's arm, or skipped."""
    _log_counts(reading, run, level)
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
                "beside skip arm=%s against=%s run=%s level=%s source=%s measure=%s scope=%s "
                "reason=%s defined=%d",
                arm,
                against,
                run,
                level,
                series.source,
                series.measure,
                series.scope,
                series.reason,
                series.defined,
            )
            continue
        logger.info(
            "beside arm=%s against=%s run=%s level=%s source=%s measure=%s scope=%s mean=%.6f "
            "low=%.6f high=%.6f holdouts=%d differing=%d undefined=%d",
            arm,
            against,
            run,
            level,
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


def _log_unjudged(rows: Sequence[Score], experiment: Experiment, candidate: str) -> None:
    """Log each interval every criterion of `CRITERION_NAMES` reads, with no verdict.

    Raises:
        EvaluationError: As `judge`.
    """
    every = replace(experiment, criteria=CRITERION_NAMES)
    for c in judge(rows, every, candidate=candidate):
        logger.info(
            "unjudged criterion=%s scope=%s arm=%s against=%s measure=%s mean=%.6f low=%.6f "
            "high=%.6f refuting_high=%.6f holdouts=%d differing=%d",
            c.criterion,
            c.scope,
            c.arm,
            c.against,
            c.measure,
            c.interval.mean,
            c.interval.low,
            c.interval.high,
            c.interval.refuting_high,
            c.interval.holdouts,
            c.differing,
        )


def _log_boosted(readings: Sequence[Reported | Absent]) -> None:
    """Log each of the boosting model's contrasts, with no verdict, or the pair it skips."""
    for r in readings:
        if isinstance(r, Absent):
            logger.info(
                "contrast skip model=%s arm=%s against=%s reason=absent", r.model, r.arm, r.against
            )
            continue
        logger.info(
            "contrast model=%s criterion=%s scope=%s arm=%s against=%s measure=%s mean=%.6f "
            "low=%.6f high=%.6f refuting_high=%.6f holdouts=%d differing=%d",
            r.model,
            r.criterion,
            r.scope,
            r.arm,
            r.against,
            r.measure,
            r.interval.mean,
            r.interval.low,
            r.interval.high,
            r.interval.refuting_high,
            r.interval.holdouts,
            r.differing,
        )


def _log_validity(scores: Sequence[Score], experiment: Experiment, candidate: str) -> None:
    """Judge the candidate's validity if it is a Messages arm, and log every other arm's counts.

    Each arm's line is followed by its counts and rate per fill, with no verdict.
    """
    arms = reported(scores)
    if candidate in MESSAGES_ARMS or candidate in arms:
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
