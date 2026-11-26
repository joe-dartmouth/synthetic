"""The run every subcommand of `python -m bko.evaluate` reads, and the root its folders sit under.

`named` gives the `Context` of a named run, and `runs_root` the run folders' root a subcommand's
`--runs-dir` sets.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.named_runs import DEFAULT_RUN


def runs_root(args: argparse.Namespace, context: Context) -> Path:
    """The run folders' root: `--runs-dir` when given, else the context's."""
    return args.runs_dir if args.runs_dir is not None else context.runs_root()


def named(name: str, context: Context) -> Context:
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
