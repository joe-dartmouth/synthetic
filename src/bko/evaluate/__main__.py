"""Command line for the evaluation: `python -m bko.evaluate score` and `compare`.

`score` fetches UCI Heart Disease into the data root if it is absent, runs every arm of E1 on each
of its holdouts and writes the scores to stdout as CSV. `compare SCORES` reads that CSV and writes
every contrast of E1's criteria, with each criterion's verdict, to stdout as CSV; `--candidate`
judges another arm, a dry run whose verdict is not E1's. Logs go to stderr. `docs/DEVELOPMENT.md`
(Evaluation) is the usage home.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run; 3 the
data, a method's output or a scores file contradicts its contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from bko.baselines.errors import BaselineError
from bko.baselines.methods import Runner
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.uci import load_uci
from bko.evaluate.classifier import Factory, build
from bko.evaluate.criteria import judge, table
from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.experiment import E1, Experiment
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import parse, render


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
)
"""The distributions whose installed versions a `score` run logs."""


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        experiment: The configuration `score` runs and `compare` judges by.
        arms: Each arm's runner.
        factory: Builds the unfitted classifier.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    experiment: Experiment = E1
    arms: Mapping[str, Runner] = ARMS
    factory: Factory = build


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
    except (EvaluationError, BaselineError, DatasetError) as exc:
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
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _score(args: argparse.Namespace, context: Context) -> str:
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    ensure(context.uci, data_dir, open_url=context.open_url)
    frame = load_uci(data_dir, source=context.uci)
    experiment = context.experiment
    logger.info(
        "experiment holdouts=%d arms=%s replica=%s fill_to=%d",
        len(experiment.holdout_seeds),
        ",".join(experiment.arms),
        ",".join(experiment.replica_arms),
        experiment.fill_to,
    )
    logger.info("versions %s", json.dumps(_versions(), sort_keys=True))
    text = render(run(frame, experiment, context.arms, factory=context.factory))
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("scores rows=%d sha256=%s", text.count("\n") - 1, digest)
    return text


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
    contrasts = judge(parse(text), context.experiment, candidate=candidate)
    for criterion in dict.fromkeys(contrast.criterion for contrast in contrasts):
        outcome = next(c.verdict for c in contrasts if c.criterion == criterion)
        logger.info("criterion=%s verdict=%s", criterion, outcome)
    return table(contrasts)


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
    comparing = commands.add_parser("compare", help="judge a candidate against the criteria")
    comparing.add_argument("scores", type=Path, help="a CSV that `score` wrote")
    comparing.add_argument(
        "--candidate", default=None, help="the row-adding arm judged; default: the experiment's own"
    )
    return parser


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
