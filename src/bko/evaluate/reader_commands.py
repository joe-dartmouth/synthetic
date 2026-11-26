"""The reader subcommands of `python -m bko.evaluate`: each reads a recorded run's files.

`bko.evaluate.__main__` dispatches `shared`, `rows`, `panel`, `privacy`, `truth` and
`contradictions` to the handlers here; `_named` and `_runs_root` are the helpers every
subcommand shares.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bko.evaluate.added_contradictions import run_added
from bko.evaluate.context import Context
from bko.evaluate.contradictions import run as census
from bko.evaluate.data import Dataset, dataset_releases, load
from bko.evaluate.donor_attacks import DONOR_ARMS
from bko.evaluate.errors import EvaluationError
from bko.evaluate.membership import privacy
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.panel import panel
from bko.evaluate.provenance import log_versions
from bko.evaluate.rows import creatable, store
from bko.evaluate.rows_read import read_rows
from bko.evaluate.run_folder import read_folder
from bko.evaluate.scoring import data_dir
from bko.evaluate.shared import rescore
from bko.evaluate.truth import truth
from bko.gaps.scope import load_scope


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
    """Read the run's privacy from its `--rows` stores, whose folders sit under the runs root.

    A run with an arm of `DONOR_ARMS` loads the pinned releases of the data's mapped concepts,
    which its fills are re-drawn under.
    """
    log_versions()
    named = _named(args.run, context)
    root = data_dir(args)
    data = load(named.dataset, root, named)
    ontologies = None
    if DONOR_ARMS & set(named.experiment.arms):
        concepts = {row.concept for row in data.field_rows}
        releases = dataset_releases(named.dataset, named)
        ontologies, _ = load_scope(concepts, root, releases=releases, open_url=named.open_url)
    return privacy(
        args.run,
        args.stores,
        experiment=named.experiment,
        data=data,
        runs=named.runs,
        root=_runs_root(args, context),
        ontologies=ontologies,
    )


def _truth(args: argparse.Namespace, context: Context) -> str:
    """Read the run's fills against its removed records from its `--rows` stores."""
    log_versions()
    named = _named(args.run, context)
    data = load(named.dataset, data_dir(args), named)
    return truth(
        args.run,
        args.stores,
        experiment=named.experiment,
        data=data,
        runs=named.runs,
        root=_runs_root(args, context),
    )


def _contradictions(args: argparse.Namespace, context: Context) -> str:
    """The census, or with `--rows` the named run's added rows, loading its data's releases.

    Raises:
        EvaluationError: If a run comes with `--dataset`, or a run or `--runs-dir` without
            `--rows`.
    """
    if args.datasets is not None and args.run is not None:
        msg = (
            f"contradictions --dataset reads the census of the named frames, which takes no run, "
            f"not {args.run!r}; a run's added rows are read with --rows"
        )
        raise EvaluationError(msg)
    if args.stores is None:
        if args.run is not None or args.runs_dir is not None:
            msg = "contradictions reads a run's added rows from its stores, so it needs --rows"
            raise EvaluationError(msg)
        return census(data_dir(args), context, args.datasets)
    log_versions()
    run = args.run or DEFAULT_RUN
    named = _named(run, context)
    root = data_dir(args)
    data = load(named.dataset, root, named)
    concepts = {row.concept for row in data.field_rows}
    releases = dataset_releases(named.dataset, named)
    ontologies, _ = load_scope(concepts, root, releases=releases, open_url=named.open_url)
    return run_added(
        run,
        args.stores,
        data=data,
        ontologies=ontologies,
        experiment=named.experiment,
        runs=named.runs,
        root=_runs_root(args, context),
    )


def _runs_root(args: argparse.Namespace, context: Context) -> Path:
    """The run folders' root: `--runs-dir` when given, else the context's."""
    return args.runs_dir if args.runs_dir is not None else context.runs_root()


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
