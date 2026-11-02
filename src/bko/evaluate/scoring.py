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

from bko.datasets.fetch import default_data_dir
from bko.evaluate.context import Context
from bko.evaluate.data import Data, field_rows, load
from bko.evaluate.direct import run_direct
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, MESSAGES_ARMS
from bko.evaluate.guided import Knowledge
from bko.evaluate.guided_run import guided_arms
from bko.evaluate.holdout_candidates import CandidateFile, Origin
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.provenance import log_experiment
from bko.evaluate.run import run
from bko.evaluate.scores import Score, render
from bko.gaps.scope import load_scope
from bko.gaps.shares import nested
from bko.generate.flight import Limits
from bko.generate.local import SLOTS
from bko.generate.replay import Answers, Replay, Resume
from bko.generate.weights import LOCAL_MODELS


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
    """Run every arm of `context`'s experiment on its dataset and return the scores as CSV.

    The run is `context`'s experiment, settings and dataset. `_prepare` refuses the flags and arms
    the run cannot use and reads what answers its calls, before any download; then the data is
    loaded once (`bko.evaluate.data.load`) and `_score` scores every arm on it. `SIGTERM` and
    `SIGHUP` interrupt it as `SIGINT` does (`interrupted_by_termination`).

    Raises:
        EvaluationError: As `_prepare`, or as the run.
        FileExistsError: If the call log or candidates file exists.
        KeyboardInterrupt: On `SIGINT`, `SIGTERM` or `SIGHUP`.
    """
    named = NamedRun(context.experiment, context.settings, context.dataset)
    with interrupted_by_termination():
        answers = _prepare(args, named, context)
        data = load(named.dataset, data_dir(args), context)
        return _score(data, named, args, context, answers)


def _arms(named: NamedRun, context: Context) -> tuple[list[str], bool]:
    """The BKO arms the run builds, and whether it builds the direct arm."""
    experiment = named.experiment
    guided = [
        name for name in experiment.arms if name in MESSAGES_ARMS and name not in context.arms
    ]
    return guided, DIRECT_ARM in experiment.arms and DIRECT_ARM not in context.arms


def _prepare(args: argparse.Namespace, named: NamedRun, context: Context) -> Answers | None:
    """Refuse the flags and arms the run cannot use, then read what answers its calls.

    It runs before any download, so a refused run fetches nothing.

    Raises:
        EvaluationError: As `_refuse_flags`, or if the direct arm runs on field rows that nest
            in a chain (`bko.gaps.shares.nested`), which it cannot draw.
        FileExistsError: If the call log or candidates file exists.
    """
    guided, direct = _arms(named, context)
    _refuse_flags(args, guided, direct=direct)
    if direct:
        try:
            nested(field_rows(named.dataset, context))
        except ValueError as exc:
            msg = f"the direct arm {DIRECT_ARM} cannot draw dataset {named.dataset}'s rows: {exc}"
            raise EvaluationError(msg) from exc
    if args.rehearse:
        # scores.csv carries no marker, so its log says what the run was before anything else.
        logger.warning(
            "rehearsal canned replies: spend, projection and stop lines are wiring checks, "
            "not paid figures"
        )
    return _answers(args, context, named.settings.model)


def data_dir(args: argparse.Namespace) -> Path:
    """The `--data-dir` given, resolved, else the repository's data root."""
    return args.data_dir.resolve() if args.data_dir is not None else default_data_dir()


def _score(
    data: Data,
    named: NamedRun,
    args: argparse.Namespace,
    context: Context,
    answers: Answers | None,
) -> str:
    """Score every arm of `named` on `data` with `answers`, once `_prepare` has checked the run.

    Every stage reads `data`: the harness splits its records under its frame's edges, the BKO arms
    observe cells by its field rows and name it in their knowledge, the rehearsal answers in its
    frame, and the experiment line, call log and candidates file name its frame. The scores' row
    count and sha256 are logged, then, once a candidates file is closed, its path, candidate lines
    and sha256.
    """
    experiment = named.experiment
    guided, direct = _arms(named, context)
    log_experiment(args.run, experiment, context.models, frame=data.frame.name)
    if guided or direct:
        scores = _guided(args, data, guided, named, context, answers, direct=direct)
    else:
        scores = run(
            data.records, experiment, context.arms, models=context.models, frame=data.frame
        )
    text = render(scores)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("scores rows=%d sha256=%s", text.count("\n") - 1, digest)
    if args.candidates is not None:
        kept = args.candidates.read_bytes()
        logger.info(
            "candidates path=%s lines=%d sha256=%s",
            args.candidates,
            len(kept.splitlines()) - 1,
            hashlib.sha256(kept).hexdigest(),
        )
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


def _answers(args: argparse.Namespace, context: Context, model: str) -> Answers | None:
    """The source that answers the BKO calls, read before any download or log is made.

    A resume's live calls go to `model`'s send (`Context.sender`).
    """
    if args.replay is not None:
        return Replay(args.replay)
    if args.resume is None:
        return None
    resume = Resume(args.resume, context.sender(model)())
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
    data: Data,
    names: Sequence[str],
    named: NamedRun,
    context: Context,
    answers: Answers | None,
    *,
    direct: bool,
) -> list[Score]:
    """Run every arm, the BKO arms `names` built by `bko.evaluate.guided_run.guided_arms`.

    With `direct`, the direct arm runs too, writing the same candidates file; with no BKO arm, it
    writes that file alone, with no settings in its header, and no call log is made. Each BKO or
    direct arm logs its run's validity sums on the way out, a stopped run included.
    """
    rows = data.field_rows
    ontologies, fragments = load_scope(
        {row.concept for row in rows},
        data_dir(args),
        releases=context.releases,
        open_url=context.open_url,
    )
    knowledge = Knowledge(data.name, ontologies, fragments, rows)
    experiment = named.experiment
    origin = Origin(args.run, data, context.releases)
    records, frame = data.records, data.frame
    if not names:
        with CandidateFile(args.candidates, origin, None) as kept:
            return run_direct(
                records,
                experiment,
                context.arms,
                knowledge,
                sink=kept,
                models=context.models,
                frame=frame,
            )
    settings = named.settings
    canned = context.rehearse(data, experiment, settings) if args.rehearse else None
    with guided_arms(
        names,
        knowledge,
        experiment,
        settings=settings,
        log=args.log,
        candidates=args.candidates,
        origin=origin,
        send=context.sender(settings.model) if canned is None else lambda: canned,
        run=args.run,
        limits=Limits(
            in_flight=_in_flight(args.in_flight, settings.model), idle=experiment.idle_limit
        ),
        answers=answers,
        rehearsal=args.rehearse,
    ) as paid:
        arms = {**context.arms, **paid.arms}
        drains = {name: arm.drain for name, arm in paid.arms.items()}
        if direct:
            return run_direct(
                records,
                experiment,
                arms,
                knowledge,
                sink=paid.candidates,
                models=context.models,
                drains=drains,
                on_holdout=paid.watch.holdout_done,
                frame=frame,
            )
        return run(
            records,
            experiment,
            arms,
            models=context.models,
            drains=drains,
            on_holdout=paid.watch.holdout_done,
            frame=frame,
        )


def _in_flight(asked: int, model: str) -> int:
    """`asked` calls in flight, or at most the server's slots for an on-device model.

    A call past the slots would queue inside the server while its idle limit ran.
    """
    if model not in LOCAL_MODELS or asked <= SLOTS:
        return asked
    logger.info("local in_flight=%d slots=%d asked=%d", SLOTS, SLOTS, asked)
    return SLOTS
