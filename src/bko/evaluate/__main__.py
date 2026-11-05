"""The evaluation's command line: `python -m bko.evaluate` and its subcommands.

The subcommands are `score`, `compare`, `merge`, `shared`, `rows`, `panel`, `privacy`,
`likelihood`, `contradictions`, `launch` and `supervise`.

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
resumes a stopped run into the new folder (`--ceiling USD` raises a ceiling or projection stop's
ceiling for that resume alone), `--in-flight N` is forwarded to `score` only when given,
and `--retries R` caps the supervisor's resumes. `supervise RUN_ID RUN` is internal to `launch`,
which starts it: it runs `score` into the claimed folder and resumes API-caused stops
(`bko.evaluate.supervise`), and exits with the last segment's code, 128 plus the signal's number
for a child a signal killed, or 2 when it or a resume cannot run.
`shared RUN --folder FOLDER [--with FOLDER]...` rescores the run in FOLDER from its own files,
less every record a `--with` folder's arms copy, to stdout (`bko.evaluate.shared`).
`rows RUN --folder FOLDER --rows PATH` rescores the run in FOLDER the same way, with no `--with`,
and stores every fill of each holdout its checks pass in PATH, which must not exist
(`bko.evaluate.rows`).
`panel RUN --folder FOLDER --rows PATH` rescores the run in FOLDER under the classifier panel,
with calibration, from the store at PATH that `rows` wrote, to stdout (`bko.evaluate.panel`).
`privacy RUN --rows PATH...` reads the membership attack, NNDR and DCR of RUN's stores, one per
slice, whose folders sit under the runs root, to stdout, and logs the reading rule
(`bko.evaluate.membership`).
`merge RUN SCORES...` joins the scores files of holdout slices into the bytes one run of RUN writes,
to stdout (`bko.evaluate.merge`); it refuses files that share a holdout, whose holdouts are not
RUN's, whose columns differ, whose holdouts hold different scores' keys, or whose sibling
`scores.log` is missing or names other data, versions or configuration.
`contradictions` counts each data contradiction in the cells of every dataset's training split
at holdout seed 0, to stdout as CSV, with no API call (`bko.evaluate.contradictions`).
`likelihood [RUN] --candidates FILE...` reads one run's candidates files, one per slice of a sliced
run, as one, and logs G7's gate (`bko.evaluate.likelihood`).
`score --in-flight N` keeps at most N BKO calls in flight at once (`bko.generate.flight`).
`docs/DEVELOPMENT.md` (Evaluation) is the usage home.

`score --log PATH --candidates FILE` spends money: the BKO arms call the Messages API, every call
goes to the call log at PATH, and every candidate they generate, with its outcome, goes to the
candidates file FILE (`bko.evaluate.holdout_candidates`) as each holdout's gap fills and then its
replica fills are drained, so a stopped run's FILE lacks the stopped holdout's undrained fills;
neither file may exist. `score --replay LOG`
answers every BKO call from the call log at LOG instead. `score --resume STOPPED --log NEW`
finishes a stopped run: it answers the stopped log's answered calls from it and sends the rest
live, all into NEW; a ceiling or projection stop resumes only with `--ceiling USD`, the ceiling
that resume alone runs under. `score --rehearse` answers every BKO call with canned records
(`bko.evaluate.rehearse`), so the stop rules and spend lines run for free. `score g2
--candidates FILE` is free: its direct arm draws from the fused BKB (`bko.evaluate.direct`), writes
FILE, and takes no `--log`.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run,
including an existing call log or candidates file, a `rows` store path that exists or whose
directory does not, Messages arms with no `--log` or `--candidates`, a direct arm with no
`--candidates` or over field rows that nest in a chain, `--log`, `--replay`, `--resume` or
`--rehearse` with no Messages arm to build, `--ceiling` with no `--resume`, `--candidates` with no
Messages or direct arm, one path for both files, more than one of `--replay`, `--resume` and
`--rehearse`, a call with no response, a failed gap query, a run whose spend reached the
ceiling, projects to it or stalled, a `launch` on battery power or into a folder that holds a run,
a `privacy` run with no reference arm or removed cell, and scores whose validity cannot be read;
3 the data, a method's output, a scores file, a replay or a stopped log that cannot be resumed
contradicts its contract, or a `shared`, `rows`, `panel` or `privacy` run folder or store fails an
identity check.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from bko.baselines.errors import BaselineError
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.datasets.run_folders import RUN_ID, runs_root
from bko.evaluate.arguments import parser
from bko.evaluate.compare import compare
from bko.evaluate.context import Context
from bko.evaluate.contradictions import run as census
from bko.evaluate.data import Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import MESSAGES_ARMS
from bko.evaluate.holdout_candidates import HoldoutCandidates
from bko.evaluate.launch import launch
from bko.evaluate.likelihood import candidates_file, likelihood
from bko.evaluate.logs import stamped_handler
from bko.evaluate.membership import privacy
from bko.evaluate.merge import merge
from bko.evaluate.named_runs import DEFAULT_RUN, NamedRun, sliced
from bko.evaluate.panel import panel
from bko.evaluate.provenance import log_experiment
from bko.evaluate.resumes import resumable
from bko.evaluate.rows import creatable, store
from bko.evaluate.rows_read import read_rows
from bko.evaluate.run_folder import read_folder
from bko.evaluate.scoring import data_dir, score
from bko.evaluate.shared import rescore
from bko.evaluate.supervise import supervise
from bko.generate.api import HTTP_LOGGERS
from bko.generate.errors import GenerationError
from bko.generate.resume_guard import Ceiling


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
    args = parser(__doc__).parse_args(argv)
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
    """Run any subcommand but `score`, writing to `sink`."""
    if args.command == "supervise":
        return _supervise(args, context)
    if args.command == "launch":
        sink.write(_launch(args, context))
    elif args.command == "merge":
        sink.write(merge(_named(args.run, context).experiment.holdout_seeds, args.scores))
    elif args.command == "likelihood":
        _likelihood(args, context, sink)
    elif args.command == "shared":
        sink.write(_shared(args, context))
    elif args.command == "rows":
        sink.write(_rows(args, context))
    elif args.command == "panel":
        sink.write(_panel(args, context))
    elif args.command == "privacy":
        sink.write(_privacy(args, context))
    elif args.command == "contradictions":
        sink.write(census(data_dir(args), context))
    else:
        _compare(args, context, sink)
    return 0


def _launch(args: argparse.Namespace, context: Context) -> str:
    """Start the run, and return the command that watches its folder, one line.

    The named run defaults to the run id's entry, so `launch e1-4` launches `e1`.

    Raises:
        EvaluationError: If the run id is not a run id of record, the run has no Messages arm, so
            `score --log` would refuse it, `--ceiling` comes without `--resume`, or as
            `resumable` and `launch`.
    """
    match = RUN_ID.fullmatch(args.run_id)
    if match is None or match["entry"] is None:
        msg = f"{args.run_id!r} is not a run id of record, such as e1-4"
        raise EvaluationError(msg)
    name = args.run or match["entry"]
    context = _named(name, context)
    if not set(context.experiment.arms) & set(MESSAGES_ARMS):
        msg = f"the run {name!r} has no Messages arm, so it spends nothing to launch"
        raise EvaluationError(msg)
    if args.ceiling is not None and args.resume is None:
        msg = "--ceiling raises a resumed run's spend ceiling, so it needs --resume"
        raise EvaluationError(msg)
    raised = None
    if args.resume is not None and args.resume.is_file():
        holdouts = len(context.experiment.holdout_seeds)
        ceiling = None if args.ceiling is None else Ceiling(args.ceiling, holdouts)
        raised = resumable(args.resume, context.settings, ceiling, run=name)
    found = launch(
        args.run_id,
        name,
        root=context.runs_dir if context.runs_dir is not None else runs_root(),
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        rehearse=args.rehearse,
        resume=args.resume,
        raised=raised,
        in_flight=args.in_flight,
        retries=args.retries,
        power=context.power,
        start=context.start,
    )
    logger.info("launch started=%s pid=%s command=%s", found.started, found.pid, found.command)
    return f"{found.watch}\n"


def _supervise(args: argparse.Namespace, context: Context) -> int:
    """Run the launched run's segments; return the last one's exit code."""
    named = _named(args.run, context)
    return supervise(
        args.run_id,
        args.run,
        settings=named.settings,
        holdouts=len(named.experiment.holdout_seeds),
        root=args.runs_dir,
        data_dir=args.data_dir,
        rehearse=args.rehearse,
        resume=args.resume,
        ceiling=args.ceiling,
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


def _likelihood(args: argparse.Namespace, context: Context, sink: TextIO) -> None:
    """Score the run's candidates by their BKB likelihood, write the per-cell CSV, log the gate.

    Each `--candidates` file is read and checked in turn, as the gate consumes it, so one file's
    lines are held at a time. A single file's `run` must be the named run; a sliced run's files
    name their slices' runs, each a registered run with the named run's configuration but for its
    spend ceiling, over holdouts among the named run's (`_slice`).

    Raises:
        EvaluationError: If a single file is another run's, a sliced run's file is a run that is
            not a slice of the named run, a schema-1 file is read for a run of other data, or a
            schema-2 or schema-3 file's data digest is not the run's data's.
        CandidatesFormatError: If a file is not a candidates file.
    """
    named = _named(args.run, context)
    paths: list[Path] = args.candidates
    single = len(paths) == 1
    pending = [candidates_file(paths[0], args.run, named.dataset, single=single)]
    data = load(named.dataset, data_dir(args), named)
    log_experiment(args.run, named.experiment, context.models, frame=data.frame.name)

    def files() -> Iterator[HoldoutCandidates]:
        for index, path in enumerate(paths):
            body, candidates = (
                pending.pop()
                if index == 0
                else candidates_file(path, args.run, named.dataset, single=single)
            )
            header = candidates.header
            if not single:
                _slice(str(header["run"]), args.run, named)
            if header["schema"] != 1 and header["data"] != data.digest:
                msg = (
                    f"the candidates file {path}'s data digest {header['data']!r} is not the "
                    f"run's {data.digest}"
                )
                raise EvaluationError(msg)
            logger.info(
                "candidates path=%s schema=%d sha256=%s",
                path,
                header["schema"],
                hashlib.sha256(body).hexdigest(),
            )
            yield candidates

    likelihood(named.experiment, data, files(), sink)


def _slice(name: str, run: str, whole: Context) -> None:
    """Refuse the run `name` unless it is a registered slice of `run`, whose context is `whole`.

    Raises:
        EvaluationError: If `name` is not registered, or is not `sliced` from `run`.
    """
    part = whole.runs.get(name)
    if part is None or not sliced(part, NamedRun(whole.experiment, whole.settings, whole.dataset)):
        msg = f"the candidates file is run {name!r}'s, which is not a slice of {run!r}"
        raise EvaluationError(msg)


def _shared(args: argparse.Namespace, context: Context) -> str:
    """Rescore the folder's run less every `--with` folder's copies; read every file first."""
    named = _named(args.run, context)
    folder, others = read_folder(args.folder), [read_folder(path) for path in args.others]
    data = load(named.dataset, data_dir(args), named)
    uci = named.dataset is Dataset.UCI
    return rescore(
        folder,
        others,
        run=args.run,
        experiment=named.experiment,
        data=data,
        arms=named.arms,
        models=named.models,
        uci=uci,
    )


def _rows(args: argparse.Namespace, context: Context) -> str:
    """Rescore the folder's run and store every checked holdout's fills at `--rows`.

    A `--rows` path that exists, or whose directory does not, is refused before any file is read.
    """
    creatable(args.rows)
    named = _named(args.run, context)
    folder = read_folder(args.folder)
    data = load(named.dataset, data_dir(args), named)
    return store(
        folder,
        args.rows,
        run=args.run,
        experiment=named.experiment,
        data=data,
        arms=named.arms,
        models=named.models,
        uci=named.dataset is Dataset.UCI,
    )


def _panel(args: argparse.Namespace, context: Context) -> str:
    """Score the folder's run under the context's panel from its `--rows` store."""
    named = _named(args.run, context)
    folder, rows = read_folder(args.folder), read_rows(args.rows)
    data = load(named.dataset, data_dir(args), named)
    return panel(
        folder,
        rows,
        store=args.rows,
        run=args.run,
        experiment=named.experiment,
        data=data,
        arms=named.arms,
        models=named.panel,
        uci=named.dataset is Dataset.UCI,
    )


def _privacy(args: argparse.Namespace, context: Context) -> str:
    """Read the run's privacy from its `--rows` stores, whose folders sit under the runs root."""
    named = _named(args.run, context)
    data = load(named.dataset, data_dir(args), named)
    return privacy(
        args.run,
        args.stores,
        experiment=named.experiment,
        data=data,
        runs=named.runs,
        root=context.runs_dir if context.runs_dir is not None else runs_root(),
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


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr; keep stdout for the CSV."""
    logging.basicConfig(handlers=[stamped_handler(sys.stderr)], level=logging.INFO, force=True)
    logging.captureWarnings(True)
    for name in (*QUIET_LOGGERS, *HTTP_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
