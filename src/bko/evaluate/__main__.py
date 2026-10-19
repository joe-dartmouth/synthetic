"""Command line for the evaluation: `python -m bko.evaluate score` and `compare`.

`score` fetches UCI Heart Disease into the data root if it is absent, runs every arm of E1 on each
of its holdouts and writes the scores to stdout as CSV. `compare SCORES` reads that CSV and writes
every contrast of E1's criteria, with each criterion's verdict, to stdout as CSV; `--candidate`
judges another arm, a dry run whose verdict is not E1's. Logs go to stderr. `docs/DEVELOPMENT.md`
(Evaluation) is the usage home.

`score --log PATH` spends money: the BKO arms call the Messages API, and every call goes to the
call log at PATH, which must not exist.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run,
including an existing call log, BKO arms with no `--log`, a call with no response, a failed gap
query and a run whose spend reached the ceiling; 3 the data, a method's output or a scores file
contradicts its contract.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.methods import Runner
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.uci import load_uci
from bko.evaluate.classifier import Factory, build
from bko.evaluate.criteria import JUDGED, judge, table
from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.experiment import BKO_ARMS, E1, Experiment
from bko.evaluate.guided import GuidedArm, Knowledge
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import Measure, Scope, Score, parse, render
from bko.evaluate.scores import Source as ScoreSource
from bko.evaluate.validity import reported, validity, verdict
from bko.gaps.scope import load_scope
from bko.generate.api import HTTP_LOGGERS, default_send
from bko.generate.errors import GenerationError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Send
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.evaluate")

VERSIONED: Final = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "sdv",
    "sdmetrics",
    "torch",
    "anthropic",
    "pronto",
)
"""The distributions whose installed versions a `score` run logs."""


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        experiment: The configuration `score` runs and `compare` judges by.
        arms: Each arm's runner; a BKO name present here runs as given, so tests can stand in for
            the API.
        factory: Builds the unfitted classifier.
        releases: The ontology releases the BKO arms' gap detection reads, by concept prefix.
        field_rows: The field mappings the BKO arms observe cells by.
        send: Builds the send every BKO arm's call goes through.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    experiment: Experiment = E1
    arms: Mapping[str, Runner] = ARMS
    factory: Factory = build
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    send: Callable[[], Send] = default_send


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run one subcommand and return the exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the CSV goes; stdout when omitted. Nothing is written unless the run succeeds.

    Returns:
        0 done, 2 the command could not run, 3 an input contradicts its contract. Any other
        exception propagates, so the process exits 1 with its traceback.
    """
    context = context or Context()
    args = _parser().parse_args(argv)
    try:
        text = _score(args, context) if args.command == "score" else _compare(args, context)
    except (EvaluationError, BaselineError, DatasetError, GenerationError, BKBError) as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    (out or sys.stdout).write(text)
    return 0


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr; keep stdout for the CSV."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True)
    for name in (*QUIET_LOGGERS, *HTTP_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)


def _score(args: argparse.Namespace, context: Context) -> str:
    experiment = context.experiment
    guided = [name for name in experiment.arms if name in BKO_ARMS and name not in context.arms]
    if guided and args.log is None:
        msg = f"--log is required: the arms {guided} call the Messages API"
        raise EvaluationError(msg)
    if args.log is not None and args.log.exists():
        raise FileExistsError(errno.EEXIST, "the call log exists", str(args.log))
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    ensure(context.uci, data_dir, open_url=context.open_url)
    frame = load_uci(data_dir, source=context.uci)
    _log_experiment(experiment)
    if guided:
        scores = _guided(args, data_dir, frame, guided, context)
    else:
        scores = run(frame, experiment, context.arms, factory=context.factory)
    text = render(scores)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("scores rows=%d sha256=%s", text.count("\n") - 1, digest)
    return text


def _log_experiment(experiment: Experiment) -> None:
    logger.info(
        "experiment holdouts=%d arms=%s replica=%s fill_to=%d budget=%d validity=%s "
        "spend_ceiling=%.2f",
        len(experiment.holdout_seeds),
        ",".join(experiment.arms),
        ",".join(experiment.replica_arms),
        experiment.fill_to,
        experiment.budget,
        experiment.validity,
        experiment.spend_ceiling,
    )
    logger.info("versions %s", json.dumps(_versions(), sort_keys=True))


def _guided(
    args: argparse.Namespace,
    data_dir: Path,
    frame: pd.DataFrame,
    names: Sequence[str],
    context: Context,
) -> list[Score]:
    """Run every arm, building a `GuidedArm` for each of `names` over one shared call log.

    Each BKO arm logs its run's validity sums on the way out, a stopped run included.
    """
    rows = context.field_rows
    ontologies, fragments = load_scope(
        {row.concept for row in rows},
        data_dir,
        releases=context.releases,
        open_url=context.open_url,
    )
    knowledge = Knowledge(context.uci.name, ontologies, fragments, rows)
    experiment = context.experiment
    with CallLog(args.log, SETTINGS) as log:
        send = context.send()
        arms = {
            name: GuidedArm(
                name,
                BKO_ARMS[name],
                knowledge,
                send=send,
                log=log,
                budget=experiment.budget,
                ceiling=experiment.spend_ceiling,
            )
            for name in names
        }
        try:
            return run(
                frame,
                experiment,
                {**context.arms, **arms},
                factory=context.factory,
                drains={name: arm.drain for name, arm in arms.items()},
            )
        finally:
            for arm in arms.values():
                arm.summary()


def _compare(args: argparse.Namespace, context: Context) -> str:
    data = args.scores.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"the scores are not UTF-8: {exc}"
        raise ScoresFormatError(msg) from exc
    candidate = context.experiment.candidate if args.candidate is None else args.candidate
    logger.info(
        "compare candidate=%s dry_run=%s scores_sha256=%s",
        candidate,
        candidate != context.experiment.candidate,
        hashlib.sha256(data).hexdigest(),
    )
    scores = parse(text)
    contrasts = judge(scores, context.experiment, candidate=candidate)
    for criterion in dict.fromkeys(contrast.criterion for contrast in contrasts):
        outcome = next(c.verdict for c in contrasts if c.criterion == criterion)
        logger.info("criterion=%s verdict=%s", criterion, outcome)
    _log_validity(scores, context.experiment, candidate)
    _log_copies(scores)
    _log_means(scores)
    return table(contrasts)


def _log_validity(scores: Sequence[Score], experiment: Experiment, candidate: str) -> None:
    """Judge the candidate's validity if it is a BKO arm, and log every other arm's counts."""
    arms = reported(scores)
    if candidate in BKO_ARMS or candidate in arms:
        found = validity(scores, experiment, candidate)
        logger.info("criterion=validity verdict=%s %s", verdict(found, experiment), found.line())
    for arm in arms:
        if arm != candidate:
            logger.info("validity %s", validity(scores, experiment, arm).line())


def _log_copies(scores: Sequence[Score]) -> None:
    """Log each arm's test records copied, summed over the holdouts, by added and replica rows."""
    totals: dict[str, dict[ScoreSource, float]] = {}
    for s in scores:
        if s.measure is Measure.TEST_COPIES:
            found = totals.setdefault(s.arm, {})
            found[s.source] = found.get(s.source, 0.0) + s.value
    for arm, found in totals.items():
        added, replica = found.get(ScoreSource.ADDED, 0.0), found.get(ScoreSource.REPLICA, 0.0)
        logger.info("copies arm=%s added=%d replica=%d", arm, added, replica)


def _log_means(scores: Sequence[Score]) -> None:
    """Log each arm's mean of every judged measure in every scope over the holdouts; no verdict."""
    for arm in dict.fromkeys(s.arm for s in scores if s.source is ScoreSource.AUGMENTED):
        for measure in JUDGED:
            for scope in Scope:
                values = [
                    s.value
                    for s in scores
                    if (s.arm, s.source, s.measure, s.scope)
                    == (arm, ScoreSource.AUGMENTED, measure, scope)
                ]
                mean = sum(values) / len(values)
                logger.info(
                    "mean arm=%s measure=%s scope=%s value=%.6f holdouts=%d",
                    arm,
                    measure,
                    scope,
                    mean,
                    len(values),
                )


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version(), "platform": platform.platform()}
    for name in VERSIONED:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "absent"
    return found


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.evaluate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score", help="score every arm on every holdout")
    scoring.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    scoring.add_argument(
        "--log",
        type=Path,
        default=None,
        help="the call log to create for the BKO arms' calls; it must not exist",
    )
    comparing = commands.add_parser("compare", help="judge a candidate against the criteria")
    comparing.add_argument("scores", type=Path, help="a CSV that `score` wrote")
    comparing.add_argument(
        "--candidate", default=None, help="the row-adding arm judged; default: the experiment's own"
    )
    return parser


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
