"""Command line for the evaluation: `python -m bko.evaluate score`, `compare` and `launch`.

`score` fetches UCI Heart Disease into the data root if it is absent, runs every arm of E1 on each
of its holdouts and writes the scores to stdout as CSV. `compare SCORES` reads that CSV and writes
every contrast of E1's criteria, with each criterion's verdict, to stdout as CSV; its logs carry
each BKO arm's counts per fill and each replica arm's mean replica utility and fidelity.
`--candidate` judges another arm, a dry run whose verdict is not E1's. Logs go to stderr.
`launch RUN_ID [RUN]` starts a paid `score` detached (`bko.evaluate.launch`).
`docs/DEVELOPMENT.md` (Evaluation) is the usage home.

`score --log PATH` spends money: the BKO arms call the Messages API, and every call goes to the
call log at PATH, which must not exist. `score --replay LOG` answers every BKO call from the call
log at LOG instead. `score --resume STOPPED --log NEW` finishes a stopped run: it answers the
stopped log's answered calls from it and sends the rest live, all into NEW.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run,
including an existing call log, BKO arms with no `--log`, `--log`, `--replay` or `--resume` with
no BKO arm to build, `--resume` with `--replay`, a call with no response, a failed gap query, a
run whose spend reached the ceiling, projects to it or stalled, a `launch` on battery power
or into a folder that holds a run, and scores whose validity cannot be read; 3 the data, a
method's output, a scores file, a replay or a stopped log that cannot be resumed contradicts its
contract.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.methods import Runner
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.run_folders import runs_root
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.uci import load_uci
from bko.evaluate.classifier import Factory, build
from bko.evaluate.compare import compare
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import BKO_ARMS, DEFAULT_RUN, E1, RUNS, Experiment, NamedRun
from bko.evaluate.guided import Knowledge, guided_arms
from bko.evaluate.launch import Power, Spawn, launch, on_battery, spawn
from bko.evaluate.provenance import log_experiment
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import Score, render
from bko.gaps.scope import load_scope
from bko.generate.api import HTTP_LOGGERS, default_send
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS, Send, Settings
from bko.generate.replay import Replay, Resume
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.evaluate")


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        experiment: The configuration the run named `e1` is; every other name reads `runs`.
        settings: The generator settings of the run named `e1`.
        runs: Every other named run, by name.
        arms: Each arm's runner; a BKO name present here runs as given, so tests can stand in for
            the API.
        factory: Builds the unfitted classifier.
        releases: The ontology releases the BKO arms' gap detection reads, by concept prefix.
        field_rows: The field mappings the BKO arms observe cells by.
        send: Builds the send every BKO arm's call goes through.
        power: Whether the machine runs on battery, for `launch`.
        start: Starts `launch`'s detached run.
        runs_dir: The run folders' root for `launch`; `runs/` at the repository root when None.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    experiment: Experiment = E1
    settings: Settings = SETTINGS
    runs: Mapping[str, NamedRun] = RUNS
    arms: Mapping[str, Runner] = ARMS
    factory: Factory = build
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    send: Callable[[], Send] = default_send
    power: Power = on_battery
    start: Spawn = spawn
    runs_dir: Path | None = None


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
        out: Where the CSV goes; stdout when omitted. `score` writes nothing unless the run
            succeeds; `compare` writes its table before it reads validity, so a validity it
            cannot read still leaves the interval criteria's table.

    Returns:
        0 done, 2 the command could not run, 3 an input contradicts its contract. Any other
        exception propagates, so the process exits 1 with its traceback.
    """
    context = context or Context()
    args = _parser().parse_args(argv)
    sink = out or sys.stdout
    try:
        if args.command == "launch":
            sink.write(_launch(args, context))
            return 0
        if args.command == "compare":
            compare(_named(args.run, context).experiment, args.candidate, args.scores, sink)
            return 0
        text = _score(args, _named(args.run, context))
    except (EvaluationError, BaselineError, DatasetError, GenerationError, BKBError) as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    sink.write(text)
    return 0


def _launch(args: argparse.Namespace, context: Context) -> str:
    """Start the run, and return the command that watches its folder, one line.

    Raises:
        EvaluationError: If the run has no BKO arm, so `score --log` would refuse it, or as
            `launch`.
    """
    context = _named(args.run, context)
    if not set(context.experiment.arms) & set(BKO_ARMS):
        msg = f"the run {args.run!r} has no BKO arm, so it spends nothing to launch"
        raise EvaluationError(msg)
    found = launch(
        args.run_id,
        args.run,
        root=context.runs_dir if context.runs_dir is not None else runs_root(),
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        power=context.power,
        start=context.start,
    )
    logger.info("launch started=%s pid=%s command=%s", found.started, found.pid, found.command)
    return f"{found.watch}\n"


def _named(name: str, context: Context) -> Context:
    """`context` with the experiment and settings of the run `name`.

    `e1` is the context's own experiment and settings, so a test can stand in for it.

    Raises:
        EvaluationError: If no run has that name.
    """
    if name == DEFAULT_RUN:
        return context
    run = context.runs.get(name)
    if run is None:
        msg = f"no run named {name!r}; the runs are {', '.join(sorted(context.runs))}"
        raise EvaluationError(msg)
    return replace(context, experiment=run.experiment, settings=run.settings)


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr; keep stdout for the CSV."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    if args.replay is not None and not guided:
        msg = "--replay needs a BKO arm"
        raise EvaluationError(msg)
    if args.resume is not None and not guided:
        msg = "--resume needs a BKO arm"
        raise EvaluationError(msg)
    if args.log is not None and not guided:
        msg = "--log needs a BKO arm"
        raise EvaluationError(msg)
    if args.log is not None and args.log.exists():
        raise FileExistsError(errno.EEXIST, "the call log exists", str(args.log))
    answers = _answers(args, context)
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    ensure(context.uci, data_dir, open_url=context.open_url)
    frame = load_uci(data_dir, source=context.uci)
    log_experiment(args.run, experiment)
    if guided:
        scores = _guided(args, data_dir, frame, guided, context, answers)
    else:
        scores = run(frame, experiment, context.arms, factory=context.factory)
    text = render(scores)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("scores rows=%d sha256=%s", text.count("\n") - 1, digest)
    return text


def _answers(args: argparse.Namespace, context: Context) -> Replay | None:
    """The source that answers the BKO calls, read before any download or log is made."""
    if args.replay is not None:
        return Replay(args.replay)
    if args.resume is None:
        return None
    resume = Resume(args.resume, context.send())
    logger.info(
        "resume log=%s sha256=%s answered=%d",
        args.resume,
        resume.digest,
        resume.answered.calls,
    )
    return resume


def _guided(
    args: argparse.Namespace,
    data_dir: Path,
    frame: pd.DataFrame,
    names: Sequence[str],
    context: Context,
    answers: Replay | None,
) -> list[Score]:
    """Run every arm, the BKO arms `names` built by `bko.evaluate.guided.guided_arms`.

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
    with guided_arms(
        names,
        knowledge,
        experiment,
        settings=context.settings,
        log=args.log,
        send=context.send,
        answers=answers,
    ) as paid:
        return run(
            frame,
            experiment,
            {**context.arms, **paid.arms},
            factory=context.factory,
            drains={name: arm.drain for name, arm in paid.arms.items()},
            on_holdout=paid.watch.holdout_done,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.evaluate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score", help="score every arm on every holdout")
    scoring.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
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
    answering = scoring.add_mutually_exclusive_group()
    answering.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="answer every BKO call from this call log instead of the API",
    )
    answering.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "finish the stopped run whose call log this is: answer its answered calls from it, "
            "then send the rest live"
        ),
    )
    launching = commands.add_parser(
        "launch", help="start a paid score run detached, kept awake and on mains power"
    )
    launching.add_argument("run_id", help="the run's id, such as e1-4; its folder under runs/")
    launching.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    launching.add_argument(
        "--data-dir", type=Path, default=None, help="the data root `score` reads"
    )
    launching.add_argument(
        "--dry-run", action="store_true", help="write the folder's roles and start nothing"
    )
    comparing = commands.add_parser("compare", help="judge a candidate against the criteria")
    comparing.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    comparing.add_argument("scores", type=Path, help="a CSV that `score` wrote")
    comparing.add_argument(
        "--candidate", default=None, help="the row-adding arm judged; default: the experiment's own"
    )
    return parser


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
