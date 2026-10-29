"""The evaluation's command line: `python -m bko.evaluate` and its subcommands.

The subcommands are `score`, `compare`, `merge`, `launch` and `supervise`.

`score` loads the named run's dataset (`bko.evaluate.data`; for UCI it fetches UCI Heart Disease
into the data root if it is absent), runs every arm of E1 on each of its holdouts and writes the
scores to stdout as CSV. `compare SCORES` reads that CSV and writes
every contrast of E1's criteria, with each criterion's verdict, to stdout as CSV; its logs carry
each BKO arm's counts per fill and each replica arm's mean replica utility and fidelity.
`--candidate` judges another arm, a dry run whose verdict is not E1's; `--beside SCORES` logs the
judged arm read beside another file's arm, with no verdict (`bko.evaluate.beside`). Logs go to
stderr.
`launch RUN_ID [RUN]` starts a paid `score` detached under `supervise` (`bko.evaluate.launch`);
RUN defaults to the run id's entry; `--rehearse` starts its rehearsal instead, `--resume STOPPED`
resumes a stopped run into the new folder, `--in-flight N` is forwarded to `score` only when given,
and `--retries R` caps the supervisor's resumes. `supervise RUN_ID RUN` is internal to `launch`,
which starts it: it runs `score` into the claimed folder and resumes API-caused stops
(`bko.evaluate.supervise`), and exits with the last segment's code, 128 plus the signal's number
for a child a signal killed, or 2 when it or a resume cannot run.
`merge RUN SCORES...` joins the scores files of holdout slices into the bytes one run of RUN writes,
to stdout (`bko.evaluate.merge`); it refuses files that share a holdout, whose holdouts are not
RUN's, whose columns differ, or whose holdouts hold different scores' keys.
`score --in-flight N` keeps at most N BKO calls in flight at once (`bko.generate.flight`).
`docs/DEVELOPMENT.md` (Evaluation) is the usage home.

`score --log PATH --candidates FILE` spends money: the BKO arms call the Messages API, every call
goes to the call log at PATH, and every candidate they generate, with its outcome, goes to the
candidates file FILE (`bko.evaluate.holdout_candidates`) as each holdout's gap fills and then its
replica fills are drained, so a stopped run's FILE lacks the stopped holdout's undrained fills;
neither file may exist. `score --replay LOG`
answers every BKO call from the call log at LOG instead. `score --resume STOPPED --log NEW`
finishes a stopped run: it answers the stopped log's answered calls from it and sends the rest
live, all into NEW. `score --rehearse` answers every BKO call with canned records
(`bko.evaluate.rehearse`), so the stop rules and spend lines run for free. `score g2
--candidates FILE` is free: its direct arm draws from the fused BKB (`bko.evaluate.direct`), writes
FILE, and takes no `--log`.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run,
including an existing call log or candidates file, BKO arms with no `--log` or `--candidates`, a
direct arm with no `--candidates`, `--log`, `--replay`, `--resume` or `--rehearse` with no BKO arm
to build, `--candidates` with no BKO or direct arm, one
path for both files, more than one of `--replay`, `--resume` and `--rehearse`, a call with no
response, a failed gap query, a run whose spend reached the ceiling, projects to it or stalled, a
`launch` on battery power or into a folder that holds a run, and scores whose validity cannot be
read; 3 the data, a method's output, a scores file, a replay or a stopped log that cannot be
resumed contradicts its contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from bko.baselines.errors import BaselineError
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.datasets.run_folders import RUN_ID, runs_root
from bko.evaluate.compare import compare
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import MESSAGES_ARMS
from bko.evaluate.launch import MAX_RETRIES, RETRIES, launch, resumable
from bko.evaluate.merge import merge
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.scoring import score
from bko.evaluate.supervise import supervise
from bko.generate.api import HTTP_LOGGERS
from bko.generate.errors import GenerationError
from bko.generate.flight import IN_FLIGHT


logger = logging.getLogger("bko.evaluate")


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
        0 done, 2 the command could not run, 3 an input contradicts its contract; `supervise`
        returns its last segment's exit code, 128 plus the signal's number for a child a signal
        killed, or 2 when it cannot run or a resume cannot. Any other exception propagates, so the
        process exits 1 with its traceback.
    """
    context = context or Context()
    args = _parser().parse_args(argv)
    sink = out or sys.stdout
    try:
        if args.command != "score":
            return _other(args, context, sink)
        text = score(args, _named(args.run, context))
    except (EvaluationError, BaselineError, DatasetError, GenerationError, BKBError) as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    sink.write(text)
    return 0


def _other(args: argparse.Namespace, context: Context, sink: TextIO) -> int:
    """Run `supervise`, `launch`, `merge` or `compare`, writing its output to `sink`."""
    if args.command == "supervise":
        return _supervise(args, context)
    if args.command == "launch":
        sink.write(_launch(args, context))
    elif args.command == "merge":
        sink.write(merge(_named(args.run, context).experiment.holdout_seeds, args.scores))
    else:
        _compare(args, context, sink)
    return 0


def _launch(args: argparse.Namespace, context: Context) -> str:
    """Start the run, and return the command that watches its folder, one line.

    The named run defaults to the run id's entry, so `launch e1-4` launches `e1`.

    Raises:
        EvaluationError: If the run id is not a run id of record, the run has no BKO arm, so
            `score --log` would refuse it, or as `launch`.
    """
    match = RUN_ID.fullmatch(args.run_id)
    if match is None or match["entry"] is None:
        msg = f"{args.run_id!r} is not a run id of record, such as e1-4"
        raise EvaluationError(msg)
    name = args.run or match["entry"]
    context = _named(name, context)
    if not set(context.experiment.arms) & set(MESSAGES_ARMS):
        msg = f"the run {name!r} has no BKO arm, so it spends nothing to launch"
        raise EvaluationError(msg)
    if args.resume is not None and args.resume.is_file():
        resumable(args.resume, context.settings)
    found = launch(
        args.run_id,
        name,
        root=context.runs_dir if context.runs_dir is not None else runs_root(),
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        rehearse=args.rehearse,
        resume=args.resume,
        in_flight=args.in_flight,
        retries=args.retries,
        power=context.power,
        start=context.start,
    )
    logger.info("launch started=%s pid=%s command=%s", found.started, found.pid, found.command)
    return f"{found.watch}\n"


def _supervise(args: argparse.Namespace, context: Context) -> int:
    """Run the launched run's segments; return the last one's exit code."""
    return supervise(
        args.run_id,
        args.run,
        settings=_named(args.run, context).settings,
        root=args.runs_dir,
        data_dir=args.data_dir,
        rehearse=args.rehearse,
        resume=args.resume,
        in_flight=args.in_flight,
        retries=args.retries,
        segment=context.segment,
        sleep=context.sleep,
        power=context.power,
    )


def _compare(args: argparse.Namespace, context: Context, sink: TextIO) -> None:
    """Judge the scores of the run `args.run` and write the table to `sink`."""
    named = _named(args.run, context)
    compare(
        named.experiment,
        args.candidate,
        args.scores,
        sink,
        run=args.run,
        level=named.settings.guidance,
        beside=args.beside,
        beside_arm=args.beside_arm,
    )


def _named(name: str, context: Context) -> Context:
    """`context` with the experiment, settings and dataset of the run `name`.

    `e1` is the context's own experiment, settings and dataset, so a test can stand in for it.

    Raises:
        EvaluationError: If no run has that name.
    """
    if name == DEFAULT_RUN:
        return context
    run = context.runs.get(name)
    if run is None:
        msg = f"no run named {name!r}; the runs are {', '.join(sorted(context.runs))}"
        raise EvaluationError(msg)
    return replace(context, experiment=run.experiment, settings=run.settings, dataset=run.dataset)


def _in_flight(text: str) -> int:
    """`--in-flight`'s value: a whole number of calls, at least 1."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"the calls in flight must be a whole number of at least 1, not {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _retries(text: str) -> int:
    """`--retries`' value: a whole number of resumes, from 0 to `MAX_RETRIES`."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if not 0 <= value <= MAX_RETRIES:
        msg = (
            f"the retries must be a whole number of at least 0 and at most {MAX_RETRIES}, "
            f"not {text!r}"
        )
        raise argparse.ArgumentTypeError(msg)
    return value


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
    scoring.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="the file to create for every candidate the BKO arms generate; it must not exist",
    )
    scoring.add_argument(
        "--in-flight",
        type=_in_flight,
        default=IN_FLIGHT,
        help=f"the most BKO calls in flight at once; default: {IN_FLIGHT}",
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
    answering.add_argument(
        "--rehearse",
        action="store_true",
        help="answer every BKO call with canned records, so the stop rules run for free",
    )
    launching = commands.add_parser(
        "launch", help="start a paid score run detached, kept awake and on mains power"
    )
    launching.add_argument("run_id", help="the run's id, such as e1-4; its folder under runs/")
    launching.add_argument(
        "run", nargs="?", default=None, help="the named run; default: the run id's entry"
    )
    launching.add_argument(
        "--data-dir", type=Path, default=None, help="the data root `score` reads"
    )
    launching.add_argument(
        "--dry-run", action="store_true", help="write the folder's roles and start nothing"
    )
    launching.add_argument(
        "--in-flight",
        type=_in_flight,
        default=None,
        help="the most BKO calls in flight at once, forwarded to score; default: score's",
    )
    launching.add_argument(
        "--retries",
        type=_retries,
        default=RETRIES,
        help=(
            f"the most resumes of API-caused stops, at most {MAX_RETRIES}; 0 runs one segment; "
            f"ignored with --rehearse or --dry-run; default: {RETRIES}"
        ),
    )
    instead = launching.add_mutually_exclusive_group()
    instead.add_argument(
        "--rehearse",
        action="store_true",
        help="rehearse the run on canned replies, into the rehearsal folder inside its folder",
    )
    instead.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume the stopped run whose call log this is, into the run id's new folder",
    )
    _supervising(commands)
    merging = commands.add_parser(
        "merge", help="join holdout slices' scores into the scores one run writes"
    )
    merging.add_argument("run", help="the named run whose holdouts the slices partition")
    merging.add_argument("scores", type=Path, nargs="+", help="a CSV that `score` wrote")
    comparing = commands.add_parser("compare", help="judge a candidate against the criteria")
    comparing.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    comparing.add_argument("scores", type=Path, help="a CSV that `score` wrote")
    comparing.add_argument(
        "--candidate", default=None, help="the row-adding arm judged; default: the experiment's own"
    )
    comparing.add_argument(
        "--beside",
        type=Path,
        default=None,
        help="another scores file; log the judged arm read beside its arm, with no verdict",
    )
    comparing.add_argument(
        "--beside-arm", default=None, help="the arm of --beside read; default: the judged arm"
    )
    return parser


def _supervising(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    supervising = commands.add_parser(
        "supervise",
        help=(
            "internal to launch, which starts it: run a launched run's score segments, resuming "
            "API-caused stops"
        ),
    )
    supervising.add_argument("run_id", help="the run's id; its folder under --runs-dir")
    supervising.add_argument("run", help="the named run")
    supervising.add_argument("--runs-dir", type=Path, required=True, help="the run folders' root")
    supervising.add_argument("--data-dir", type=Path, default=None, help="the data root")
    supervising.add_argument("--in-flight", type=_in_flight, default=None, help="as score's")
    supervising.add_argument("--retries", type=_retries, default=RETRIES, help="as launch's")
    instead = supervising.add_mutually_exclusive_group()
    instead.add_argument("--rehearse", action="store_true", help="as launch's; never resumed")
    instead.add_argument("--resume", type=Path, default=None, help="as launch's")


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
