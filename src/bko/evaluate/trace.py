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

The source record ids name training records, so the output stays in its run folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import pandas as pd

from bko.baselines.request import Request, Strata, stratum_keys
from bko.datasets.split import split
from bko.evaluate.call_prompts import CALLS, Prompts, asked, read_prompts
from bko.evaluate.data import Data, Dataset, digest, load
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import (
    DIRECT_ARM,
    MESSAGES_ARMS,
    Experiment,
    gap_request,
    replica_request,
)
from bko.evaluate.holdout_candidates import KEPT, Candidate
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.prepare import prepared, split_line
from bko.evaluate.provenance import experiment_line
from bko.evaluate.redraw import Shares, redraw
from bko.evaluate.run_folder import Folder, base, check_folders, fields, read_folder
from bko.evaluate.scoring import data_dir
from bko.evaluate.shared import returned_attempt
from bko.gaps.observe import cell_key
from bko.gaps.scope import load_scope
from bko.ontology.terms import Ontology
from bko.validate.checks import validate


if TYPE_CHECKING:
    from bko.evaluate.context import Context


logger = logging.getLogger("bko.evaluate")

_SEED: Final = re.compile(r" seed=(?P<seed>\d+) ")

type Json = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Trace:
    """One record's explanation, or a baseline fill's.

    Attributes:
        holdout: The holdout's seed.
        arm: The `arm` line's arm, `-replica` ending a replica fill's.
        fill: `gap` or `replica`.
        record_id: The record's 0-based position among the fill's rows; None for a baseline.
        cell: The cell its call or draw asked for; None for a baseline.
        record: Its value per column; None for a baseline.
        outcome: Its candidates line's outcome; None for a baseline.
        failed: The checks it failed.
        request: What asked for it: a direct draw's cell, quota, attempt and seed; a Messages
            call's cell, attempt, guidance level and prompt sections; a baseline's method and seed.
        concepts: A direct record's concepts: each one's share, its parts and S-nodes, its label
            and whether it was drawn present.
        sources: A direct record's source training record id per copied field.
        fallback: A direct record's fields read from its label group's records.
        closed: A direct record's drawn concepts closed upward under `is_a`.
        call: A Messages record's call index.
        untraced: Why a Messages record's prompt is not traced, or None.
    """

    holdout: int
    arm: str
    fill: str
    record_id: int | None = None
    cell: str | None = None
    record: Json | None = None
    outcome: str | None = None
    failed: tuple[str, ...] = ()
    request: Json | None = None
    concepts: tuple[Json, ...] = ()
    sources: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    fallback: tuple[str, ...] = ()
    closed: tuple[str, ...] = ()
    call: int | None = None
    untraced: str | None = None

    def line(self) -> str:
        """The trace as one JSON line, its keys sorted, ending in a newline."""
        found = {name.name: getattr(self, name.name) for name in dataclasses.fields(self)}
        return json.dumps(found, sort_keys=True, allow_nan=False, default=_plain) + "\n"


def _plain(value: object) -> object:
    """A mapping or a tuple as JSON writes it; anything else is refused."""
    if isinstance(value, Mapping):
        return dict(value)
    msg = f"{type(value).__name__} is not JSON"
    raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class Fill:
    """The fill a trace reads: its holdout, its `arm` line and what that line names.

    Attributes:
        holdout: The holdout's seed.
        arm: The `arm` line's arm.
        line: The `arm` line.
    """

    holdout: int
    arm: str
    line: str

    @property
    def name(self) -> str:
        """The arm, without a replica fill's suffix."""
        return base(self.arm)

    @property
    def kind(self) -> str:
        """`replica` for a replica fill, else `gap`."""
        return "gap" if self.arm == self.name else "replica"

    @property
    def seed(self) -> int:
        """The seed of the attempt that returned."""
        found = _SEED.search(self.line)
        if found is None:
            msg = f"the arm line {self.line!r} names no seed"
            raise IdentityError(msg)
        return int(found["seed"])


def select(folder: Folder, *, holdout: int, arm: str) -> Fill:
    """The fill of `arm` on `holdout` that `folder` logs.

    Raises:
        EvaluationError: If the folder has no such holdout, or logs no such arm on it.
    """
    if holdout not in folder.splits:
        msg = f"holdout {holdout} is not one of {folder.path}'s {sorted(folder.splits)}"
        raise EvaluationError(msg)
    line = folder.arms.get((holdout, arm))
    if line is None:
        logged = sorted(name for seed, name in folder.arms if seed == holdout)
        msg = f"{folder.path} logs no arm {arm!r} on holdout {holdout}; its arms are {logged}"
        raise EvaluationError(msg)
    return Fill(holdout, arm, line)


def ordered(candidates: Sequence[object], request: Request, *, where: str) -> list[int]:
    """The positions of `candidates` in the order `bko.evaluate.shared.rebuilt` places their rows.

    Raises:
        IdentityError: As `rebuilt`, if one no longer passes validation or falls in no cell the
            request asks for.
    """
    frame = request.frame
    rows = validate(list(candidates), frame=frame).accepted
    if len(rows) != len(candidates):
        msg = f"{where}: {len(candidates) - len(rows)} kept candidates no longer pass validation"
        raise IdentityError(msg)
    rank = {cell: index for index, cell in enumerate(request.quota)}
    keys = stratum_keys(rows, Strata.CELL, edges=frame.edges)
    if any(key not in rank for key in keys):
        msg = f"{where}: a kept row falls in no cell its request asks for"
        raise IdentityError(msg)
    placed = sorted(zip((rank[key] for key in keys), map(int, rows.index), strict=True))
    return [position for _, position in placed]


def _kept(lines: Sequence[Candidate]) -> tuple[int, list[Candidate]]:
    attempt = returned_attempt(lines)
    return attempt, [c for c in lines if c.attempt == attempt and c.outcome == KEPT]


def trace_direct(
    fill: Fill,
    lines: Sequence[Candidate],
    train: pd.DataFrame,
    request: Request,
    *,
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
) -> list[Trace]:
    """Re-draw a `bko-direct` fill at its `arm` line's seed and explain each kept record.

    Raises:
        IdentityError: If the re-drawn rows' sha256 is not the `arm` line's, or its kept draws are
            not the candidates file's kept lines.
    """
    seeded = dataclasses.replace(request, seed=fill.seed)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag)
    logged, rebuilt = fields(fill.line)["sha256"], digest(redrawn.rows)
    kept = redrawn.kept
    attempt, stored = _kept(lines)
    same = [dict(draw.record) for draw in kept] == [line.candidate for line in stored]
    logger.log(
        logging.INFO if same and rebuilt == logged else logging.WARNING,
        "trace arm holdout=%d arm=%s arm_sha256=%s rebuilt_sha256=%s kept_lines=%s matches=%s",
        fill.holdout,
        fill.arm,
        logged,
        rebuilt,
        str(same).lower(),
        str(same and rebuilt == logged).lower(),
    )
    if rebuilt != logged:
        msg = f"holdout {fill.holdout}'s {fill.arm} fill re-drawn at seed {fill.seed} hashes to"
        msg += f" {rebuilt}, not its arm line's {logged}"
        raise IdentityError(msg)
    if not same:
        msg = f"holdout {fill.holdout}'s {fill.arm} re-drawn records are not its kept lines"
        raise IdentityError(msg)
    facts = Shares(train, seeded.quota, data=data, ontologies=ontologies)
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    order = ordered([draw.record for draw in kept], seeded, where=where)
    return [
        Trace(
            fill.holdout,
            fill.arm,
            fill.kind,
            record_id,
            stored[k].cell,
            dict(kept[k].record),
            stored[k].outcome,
            stored[k].failed,
            {
                "attempt": attempt,
                "cell": cell_key(kept[k].cell),
                "quota": seeded.quota[kept[k].cell],
                "rows": seeded.rows,
                "seed": fill.seed,
            },
            facts.concepts(kept[k]),
            dict(kept[k].sources),
            tuple(sorted(kept[k].fallback)),
            tuple(sorted(kept[k].closed)),
        )
        for record_id, k in enumerate(order)
    ]


def trace_messages(
    fill: Fill, lines: Sequence[Candidate], request: Request, *, calls: Path
) -> list[Trace]:
    """Explain each kept record of a Messages arm's fill by its call and its prompt's sections.

    A record the log at `calls` cannot trace, every record when there is no log, is `untraced`
    with its reason (`bko.evaluate.call_prompts.asked`); the `trace partial` WARNING counts them.

    Raises:
        IdentityError: As `ordered`.
        EvaluationError: As `bko.evaluate.call_prompts.read_prompts`.
    """
    attempt, kept = _kept(lines)
    candidates = [line.candidate for line in kept]
    order = ordered(candidates, request, where=f"holdout {fill.holdout}'s {fill.arm} fill")
    wanted = {line.call for line in kept if line.call is not None}
    prompts = read_prompts(calls, wanted) if calls.is_file() else None
    rows = validate(candidates, frame=request.frame).accepted
    values = {int(at): row for at, row in zip(rows.index, rows.to_dict("records"), strict=True)}
    traces = []
    for record_id, k in enumerate(order):
        line = kept[k]
        request_of, untraced = asked(line, prompts, {"attempt": attempt})
        record = {str(name): _number(value) for name, value in values[k].items()}
        traces.append(
            Trace(
                fill.holdout,
                fill.arm,
                fill.kind,
                record_id,
                line.cell,
                record,
                line.outcome,
                line.failed,
                request_of,
                call=line.call,
                untraced=untraced,
            )
        )
    _partial(fill, traces, prompts, wanted)
    return traces


def _partial(
    fill: Fill, traces: Sequence[Trace], prompts: Prompts | None, wanted: set[int]
) -> None:
    """Log the `trace partial` WARNING when a record is `untraced`: how many, and why."""
    reasons = [each.untraced for each in traces if each.untraced is not None]
    if not reasons:
        return
    absent = CALLS if prompts is None else ",".join(map(str, sorted(wanted - set(prompts.users))))
    logger.warning(
        "trace partial holdout=%d arm=%s untraced=%d of %d missing=%s reasons=%s",
        fill.holdout,
        fill.arm,
        len(reasons),
        len(traces),
        absent or "none",
        " | ".join(sorted(set(reasons))),
    )


def _number(value: object) -> object:
    """A frame's value as JSON writes it: numpy's ints and floats as Python's."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def trace_baseline(fill: Fill) -> list[Trace]:
    """A baseline fill's one line: its method and its seed."""
    return [
        Trace(fill.holdout, fill.arm, fill.kind, request={"method": fill.name, "seed": fill.seed})
    ]


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
    count = len(_kept(lines)[1]) if fill.name == DIRECT_ARM or fill.name in MESSAGES_ARMS else 1
    if record is not None and not 0 <= record < count:
        msg = f"record {record} is not one of the fill's {count} records"
        raise EvaluationError(msg)
    if fill.name == DIRECT_ARM:
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
    if fill.name == DIRECT_ARM:
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
