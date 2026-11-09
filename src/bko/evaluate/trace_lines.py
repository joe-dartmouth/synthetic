"""The lines `trace` writes, the fill it reads, and the arm kinds that need no re-draw.

`Trace` is one record's explanation as a sorted-key JSON line (ADR
`research/record-explainability`). `select` reads the fill of one arm on one holdout from a run
folder, and `ordered` places a fill's kept candidates in the order
`bko.evaluate.shared.rebuilt` places their rows, so record i is row i of the fill.
`trace_messages` explains a Messages arm's records by their calls and prompt sections, and
`trace_baseline` a baseline's fill by its method and seed. `bko.evaluate.trace_direct` and
`bko.evaluate.trace_perturb` explain the arms a re-draw reproduces.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Request, Strata, stratum_keys
from bko.evaluate.call_prompts import CALLS, Prompts, asked, read_prompts
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.holdout_candidates import KEPT, Candidate
from bko.evaluate.run_folder import Folder, base
from bko.evaluate.shared import returned_attempt
from bko.validate.checks import validate


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


def kept_lines(lines: Sequence[Candidate]) -> tuple[int, list[Candidate]]:
    """The fill's last returned attempt and that attempt's kept lines, in file order.

    Raises:
        IdentityError: As `bko.evaluate.shared.returned_attempt`.
    """
    attempt = returned_attempt(lines)
    return attempt, [c for c in lines if c.attempt == attempt and c.outcome == KEPT]


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
    attempt, kept = kept_lines(lines)
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
        record = {str(name): json_value(value) for name, value in values[k].items()}
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


def json_value(value: object) -> object:
    """A frame's value as JSON writes it: numpy's ints and floats as Python's."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def trace_baseline(fill: Fill) -> list[Trace]:
    """A baseline fill's one line: its method and its seed."""
    return [
        Trace(fill.holdout, fill.arm, fill.kind, request={"method": fill.name, "seed": fill.seed})
    ]
