"""`python -m bko.evaluate trace`: each record of one arm's fill, with the facts that determine it.

A record's explanation is the stored or re-derivable facts that determine it (ADR
`research/record-explainability`). `trace` reads one holdout's fill of one arm from a run folder
and writes one `Trace` per record as a sorted-key JSON line, in the order
`bko.evaluate.shared.rebuilt` places the fill's rows, so record i is row i of the fill. It makes
no API call.

- `bko-direct` (`trace_direct`): the fill is re-drawn at its `arm` line's seed through
  `bko.generate.direct.sample`, which draws nothing more to name its sources, and refused
  (`IdentityError`, exit 3) unless the re-drawn rows hash to the line's sha256. Each record names
  its request's cell and quota, each concept's share with its parts and S-nodes, the training record
  each field was read from, its fallback fields and its closure; labels come from the pinned
  releases.
- A Messages arm (`trace_messages`): each record names its call and the sections of that call's
  prompt in the folder's `calls.jsonl` (`bko.evaluate.call_prompts`), under the log's guidance
  level. A record the log cannot trace is `untraced`, with the reason, and the `trace partial`
  WARNING counts them.
- A baseline (`trace_baseline`): one line naming its method and its seed.
- `bko-direct-pram` and `bko-direct-perturbed` (`trace_perturbation`): the fill's rows are
  rebuilt from its candidate lines and refused unless they hash to the `arm` line's sha256; the
  draw underneath is re-drawn, and each record adds its line's `perturb` result, the fields that
  differ from the draw and the draw's values for them. `PRAM_ARM` is among `DIRECT_ARMS`, so it
  is dispatched first.

`bko.evaluate.trace_lines` holds the lines, the fill and the arm kinds that need no re-draw, and
`bko.evaluate.trace_direct` the re-draw, `bko.evaluate.trace_perturb` the perturbation arms;
this module dispatches the command.

The source record ids name training records, so the output stays in its run folder.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from bko.datasets.split import split
from bko.evaluate.call_prompts import CALLS
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import (
    DIRECT_ARM,
    DIRECT_ARMS,
    MESSAGES_ARMS,
    PERTURBED_ARM,
    PRAM_ARM,
    Experiment,
    gap_request,
    replica_request,
)
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.prepare import prepared, split_line
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run_folder import BKO, Folder, check_folders, read_folder
from bko.evaluate.scoring import data_dir
from bko.evaluate.trace_direct import trace_direct
from bko.evaluate.trace_lines import Fill, kept_lines, select, trace_baseline, trace_messages
from bko.evaluate.trace_perturb import trace_perturbation
from bko.gaps.scope import load_scope
from bko.ontology.terms import Ontology


if TYPE_CHECKING:
    from bko.evaluate.context import Context


logger = logging.getLogger("bko.evaluate")


def trace(
    folder: Folder,
    fill: Fill,
    *,
    run: str,
    experiment: Experiment,
    data: Data,
    models: Sequence[str],
    uci: bool,
    ontologies: Mapping[str, Ontology] | None,
    record: int | None,
) -> str:
    """Every record of `fill`, or record `record` alone, as `Trace` lines.

    `record` is checked against the fill's kept lines before any re-draw.

    Raises:
        IdentityError: As `check_folders`, if the holdout's split is not the folder's, or as
            `trace_direct` and `trace_messages`.
        EvaluationError: If `record` is not a record of the fill, or as `trace_messages`.
    """
    logger.info(
        "trace start folder=%s run=%s holdout=%d arm=%s fill=%s seed=%d",
        folder.path,
        run,
        fill.holdout,
        fill.arm,
        fill.kind,
        fill.seed,
    )
    frame = data.frame
    current = experiment_line(run, experiment, models, frame=frame.name)
    check_folders(folder, [], run=run, current=current, data=data, uci=uci)
    parts = split(data.records, seed=fill.holdout, edges=frame.edges)
    ablated = prepared(parts, experiment, edges=frame.edges).ablated
    if split_line(ablated) != folder.splits[fill.holdout]:
        msg = (
            f"holdout {fill.holdout}'s split is not {folder.path}'s {folder.splits[fill.holdout]!r}"
        )
        raise IdentityError(msg)
    train = ablated.train
    asked = gap_request if fill.kind == "gap" else replica_request
    request = asked(train, experiment, seed=fill.holdout, frame=frame)
    lines = folder.lines.get((fill.holdout, fill.name, fill.kind), ())
    count = len(kept_lines(lines)[1]) if fill.name in BKO else 1
    if record is not None and not 0 <= record < count:
        msg = f"record {record} is not one of the fill's {count} records"
        raise EvaluationError(msg)
    if fill.name in {PRAM_ARM, PERTURBED_ARM}:
        traces = trace_perturbation(
            fill,
            lines,
            train,
            request,
            data=data,
            ontologies=ontologies or {},
            budget=experiment.budget,
        )
    elif fill.name == DIRECT_ARM:
        traces = trace_direct(
            fill,
            lines,
            train,
            request,
            data=data,
            ontologies=ontologies or {},
            budget=experiment.budget,
        )
    elif fill.name in MESSAGES_ARMS:
        traces = trace_messages(fill, lines, request, calls=folder.path / CALLS)
    else:
        traces = trace_baseline(fill)
    if record is not None:
        traces = [traces[record]]
    text = "".join(each.line() for each in traces)
    logger.info(
        "trace rows=%d sha256=%s folder=%s holdout=%d arm=%s record=%s untraced=%d",
        len(traces),
        hashlib.sha256(text.encode()).hexdigest(),
        folder.path,
        fill.holdout,
        fill.arm,
        "all" if record is None else record,
        sum(each.untraced is not None for each in traces),
    )
    return text


def run_trace(args: argparse.Namespace, named: Context) -> str:
    """The `trace` command over `named`, the context of the run `args.run`.

    The folder and the selection are read before any data is loaded.
    """
    folder = read_folder(args.folder)
    fill = select(folder, holdout=args.holdout, arm=args.arm)
    root = data_dir(args)
    data = load(named.dataset, root, named)
    ontologies = None
    if fill.name in {*DIRECT_ARMS, PERTURBED_ARM}:
        concepts = {row.concept for row in data.field_rows}
        ontologies, _ = load_scope(concepts, root, releases=named.releases, open_url=named.open_url)
    return trace(
        folder,
        fill,
        run=args.run,
        experiment=named.experiment,
        data=data,
        models=[model.value for model in named.models],
        uci=named.dataset is Dataset.UCI,
        ontologies=ontologies,
        record=args.record,
    )


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `trace` subcommand to `commands`."""
    tracing = commands.add_parser(
        "trace", help="explain each record of one arm's fill of one holdout, with no API call"
    )
    tracing.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    tracing.add_argument("--folder", type=Path, required=True, help="the run's folder under runs/")
    tracing.add_argument("--holdout", type=int, required=True, help="the holdout's seed")
    tracing.add_argument(
        "--arm", required=True, help="the arm line's arm; `<arm>-replica` for a replica fill"
    )
    tracing.add_argument(
        "--record", type=int, default=None, help="one record's 0-based position; default: all"
    )
    tracing.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
