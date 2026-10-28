"""`python -m bko.evaluate score`: refuse the flags a run cannot use, then score every arm.

`bko.evaluate.__main__` parses the command and maps its errors to exit codes; its docstring holds
the command's contract. While it runs, `SIGTERM` and `SIGHUP` raise `KeyboardInterrupt` as `SIGINT`
does, so a `score` an operator or a supervisor stops ends its call log `interrupted`
(`bko.evaluate.supervise`).
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import logging
import signal
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.fetch import default_data_dir, ensure
from bko.datasets.uci import load_uci
from bko.evaluate.context import Context
from bko.evaluate.direct import run_direct
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import BKO_ARMS, DIRECT_ARM
from bko.evaluate.guided import Knowledge, guided_arms
from bko.evaluate.holdout_candidates import CandidateFile, Origin
from bko.evaluate.provenance import log_experiment
from bko.evaluate.run import run
from bko.evaluate.scores import Score, render
from bko.gaps.scope import load_scope
from bko.generate.flight import Limits
from bko.generate.replay import Answers, Replay, Resume


logger = logging.getLogger("bko.evaluate")

TERMINATING: Final = (signal.SIGTERM, signal.SIGHUP)
"""The signals `score` turns into `KeyboardInterrupt`, beside `SIGINT`, which already is one."""


@contextmanager
def interrupted_by_termination() -> Generator[None]:
    """Raise `KeyboardInterrupt` on `TERMINATING` while the body runs, then restore the handlers.

    Off the main thread, where no handler can be set, it changes nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {number: signal.signal(number, signal.default_int_handler) for number in TERMINATING}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, signal.SIG_DFL if handler is None else handler)


def score(args: argparse.Namespace, context: Context) -> str:
    """Run every arm of `context`'s experiment and return the scores as CSV.

    `SIGTERM` and `SIGHUP` interrupt it as `SIGINT` does (`interrupted_by_termination`).

    Raises:
        EvaluationError: As `_refuse_flags`, or as the run.
        FileExistsError: If the call log or candidates file exists.
        KeyboardInterrupt: On `SIGINT`, `SIGTERM` or `SIGHUP`.
    """
    with interrupted_by_termination():
        return _score(args, context)


def _score(args: argparse.Namespace, context: Context) -> str:
    """Run every arm of `context`'s experiment and return the scores as CSV, as `score`."""
    experiment = context.experiment
    guided = [name for name in experiment.arms if name in BKO_ARMS and name not in context.arms]
    direct = DIRECT_ARM in experiment.arms and DIRECT_ARM not in context.arms
    _refuse_flags(args, guided, direct=direct)
    if args.rehearse:
        # scores.csv carries no marker, so its log says what the run was before anything else.
        logger.warning(
            "rehearsal canned replies: spend, projection and stop lines are wiring checks, "
            "not paid figures"
        )
    answers = _answers(args, context)
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    ensure(context.uci, data_dir, open_url=context.open_url)
    frame = load_uci(data_dir, source=context.uci)
    log_experiment(args.run, experiment, context.models)
    if guided or direct:
        scores = _guided(args, data_dir, frame, guided, context, answers, direct=direct)
    else:
        scores = run(frame, experiment, context.arms, models=context.models)
    text = render(scores)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("scores rows=%d sha256=%s", text.count("\n") - 1, digest)
    return text


def _refuse_flags(args: argparse.Namespace, guided: Sequence[str], *, direct: bool) -> None:
    """Refuse flags the run's arms cannot use, before any download or file is made.

    Raises:
        EvaluationError: As `bko.evaluate.__main__`'s exit code 2 lists.
        FileExistsError: If the call log or candidates file exists.
    """
    if guided and args.log is None:
        msg = f"--log is required: the arms {guided} call the Messages API"
        raise EvaluationError(msg)
    if args.replay is not None and not guided:
        msg = "--replay needs a BKO arm"
        raise EvaluationError(msg)
    if args.resume is not None and not guided:
        msg = "--resume needs a BKO arm"
        raise EvaluationError(msg)
    if args.rehearse and not guided:
        msg = "--rehearse needs a BKO arm"
        raise EvaluationError(msg)
    generating = [*guided, *([DIRECT_ARM] if direct else [])]
    if generating and args.candidates is None:
        msg = f"--candidates is required: the arms {generating} generate candidates"
        raise EvaluationError(msg)
    if args.log is not None and not guided:
        msg = "--log needs a BKO arm that calls the Messages API"
        raise EvaluationError(msg)
    if args.candidates is not None and not generating:
        msg = "--candidates needs a BKO arm or the direct arm"
        raise EvaluationError(msg)
    if args.log is not None and args.candidates.resolve() == args.log.resolve():
        msg = f"--candidates and --log name one file, {args.log}"
        raise EvaluationError(msg)
    if args.log is not None and args.log.exists():
        raise FileExistsError(errno.EEXIST, "the call log exists", str(args.log))
    if args.candidates is not None and args.candidates.exists():
        raise FileExistsError(errno.EEXIST, "the candidates file exists", str(args.candidates))


def _answers(args: argparse.Namespace, context: Context) -> Answers | None:
    """The source that answers the BKO calls, read before any download or log is made."""
    if args.replay is not None:
        return Replay(args.replay)
    if args.resume is None:
        return None
    resume = Resume(args.resume, context.send())
    logger.info(
        "resume log=%s sha256=%s answered=%d dropped=%d carried_usd=%.4f",
        args.resume,
        resume.digest,
        resume.answered.calls,
        resume.dropped,
        resume.answered.carried_usd,
    )
    return resume


def _guided(
    args: argparse.Namespace,
    data_dir: Path,
    frame: pd.DataFrame,
    names: Sequence[str],
    context: Context,
    answers: Answers | None,
    *,
    direct: bool,
) -> list[Score]:
    """Run every arm, the BKO arms `names` built by `bko.evaluate.guided.guided_arms`.

    With `direct`, the direct arm runs too, writing the same candidates file; with no BKO arm, it
    writes that file alone and no call log is made. Each BKO or direct arm logs its run's validity
    sums on the way out, a stopped run included.
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
    origin = Origin(args.run, context.uci, context.releases)
    if not names:
        with CandidateFile(args.candidates, origin, context.settings) as kept:
            return run_direct(
                frame, experiment, context.arms, knowledge, sink=kept, models=context.models
            )
    canned = context.rehearse(frame, experiment, context.settings) if args.rehearse else None
    with guided_arms(
        names,
        knowledge,
        experiment,
        settings=context.settings,
        log=args.log,
        candidates=args.candidates,
        origin=origin,
        send=context.send if canned is None else lambda: canned,
        run=args.run,
        limits=Limits(in_flight=args.in_flight, idle=experiment.idle_limit),
        answers=answers,
        rehearsal=args.rehearse,
    ) as paid:
        arms = {**context.arms, **paid.arms}
        drains = {name: arm.drain for name, arm in paid.arms.items()}
        if direct:
            return run_direct(
                frame,
                experiment,
                arms,
                knowledge,
                sink=paid.candidates,
                models=context.models,
                drains=drains,
                on_holdout=paid.watch.holdout_done,
            )
        return run(
            frame,
            experiment,
            arms,
            models=context.models,
            drains=drains,
            on_holdout=paid.watch.holdout_done,
        )
