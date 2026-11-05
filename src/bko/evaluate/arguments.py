"""`python -m bko.evaluate`'s arguments: every subcommand's flags and the checks on their values.

`bko.evaluate.__main__` parses with `parser` and holds the commands' contract.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from bko.evaluate.contradictions import add_parser as add_contradictions
from bko.evaluate.launch import MAX_RETRIES, RETRIES
from bko.evaluate.membership import add_parser as add_privacy
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.panel import add_parser as add_panel
from bko.evaluate.rows import add_parser as add_rows
from bko.evaluate.shared import add_parser
from bko.generate.flight import IN_FLIGHT


CEILING_HELP = (
    "with --resume of a ceiling or projection stop: the spend ceiling in USD this resume alone "
    "runs under, above the stopped run's projection"
)
"""What `--ceiling` says it does, for each subcommand that takes it."""


def parser(description: str | None) -> argparse.ArgumentParser:
    """`python -m bko.evaluate`'s parser, its subcommands' arguments and their checks."""
    parser = argparse.ArgumentParser(prog="python -m bko.evaluate", description=description)
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
        type=in_flight,
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
    scoring.add_argument("--ceiling", type=ceiling, default=None, help=CEILING_HELP)
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
        type=in_flight,
        default=None,
        help="the most BKO calls in flight at once, forwarded to score; default: score's",
    )
    launching.add_argument(
        "--retries",
        type=retries,
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
    launching.add_argument("--ceiling", type=ceiling, default=None, help=CEILING_HELP)
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
    add_parser(commands)
    add_rows(commands)
    add_panel(commands)
    add_privacy(commands)
    add_contradictions(commands)
    liking = commands.add_parser(
        "likelihood", help="score a run's candidates by their BKB likelihood and read G7's gate"
    )
    liking.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    liking.add_argument(
        "--candidates",
        type=Path,
        action="append",
        required=True,
        help="the run's holdout-candidates.jsonl; repeat it for each slice of a sliced run",
    )
    liking.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
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
    supervising.add_argument("--in-flight", type=in_flight, default=None, help="as score's")
    supervising.add_argument("--retries", type=retries, default=RETRIES, help="as launch's")
    instead = supervising.add_mutually_exclusive_group()
    instead.add_argument("--rehearse", action="store_true", help="as launch's; never resumed")
    instead.add_argument("--resume", type=Path, default=None, help="as launch's")
    supervising.add_argument(
        "--ceiling", type=ceiling, default=None, help="as launch's; the first segment's alone"
    )


def in_flight(text: str) -> int:
    """`--in-flight`'s value: a whole number of calls, at least 1."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"the calls in flight must be a whole number of at least 1, not {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def retries(text: str) -> int:
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


def ceiling(text: str) -> float:
    """`--ceiling`'s value: a finite spend in USD above 0."""
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        msg = f"the ceiling must be a finite number of USD above 0, not {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value
