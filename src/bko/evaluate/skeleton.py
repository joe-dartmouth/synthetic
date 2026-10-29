"""The prescribing arm's lines: drawn once per attempt from the fused BKB, queued per cell.

`bko-skeleton` is a `bko.evaluate.guided.GuidedArm` given a `prescriber`. Each attempt draws its
request's records once with `bko.generate.direct.sample` at the request's seed, the draw
`bko-direct` makes for the same request, and keeps each record's drawn concepts, label-only ones
left out, as one line. `Queues` holds each cell's lines in draw order. A wave takes each planned
call's lines from its cell's queue at planning, in plan order. Within one call a record is matched
to a line by concept set, not by position, and only a record kept in the call's own cell consumes
one (`bko.baselines.records.assess`); every other line asked goes back to the end of its cell's
queue, in call order, for the next top-up, never redrawn (ADR `research/e1-evaluation-protocol`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request
from bko.evaluate.errors import EvaluationError
from bko.gaps.observe import Cell
from bko.generate.direct import label_only, sample
from bko.generate.errors import SamplingError


if TYPE_CHECKING:
    from bko.evaluate.guided import Knowledge

type Lines = Mapping[Cell, tuple[frozenset[str], ...]]
"""Each requested cell's lines, one per requested record, in draw order."""

type Prescribe = Callable[[pd.DataFrame, Request], Lines]
"""Draw a request's lines from a holdout's training rows."""


def prescriber(knowledge: Knowledge) -> Prescribe:
    """The prescribe step over `knowledge`'s field rows, releases and dataset name.

    Raises:
        BaselineError: From the step, plain, as `bko.evaluate.direct` refuses, when no training
            record matches a draw (`SamplingError`), so the harness retries with a derived seed.
    """

    def prescribe(train: pd.DataFrame, request: Request) -> Lines:
        rows = knowledge.field_rows
        try:
            drawn = sample(
                train,
                request.quota,
                rows=rows,
                ontologies=knowledge.ontologies,
                name=knowledge.name,
                seed=request.seed,
                frame=request.frame,
            )
        except SamplingError as exc:
            raise BaselineError(str(exc)) from exc
        following = label_only(rows, frame=request.frame)
        return {
            cell: tuple(draw.concepts - following for draw in draws)
            for cell, draws in drawn.items()
        }

    return prescribe


class Queues:
    """Each cell's lines not yet consumed, in the order they are asked.

    Attributes:
        requeued: The lines given back so far.
    """

    def __init__(self, lines: Mapping[Cell, Sequence[frozenset[str]]]) -> None:
        """Queue each cell's lines in draw order."""
        self._queues = {cell: deque(held) for cell, held in lines.items()}
        self.requeued = 0

    def take(self, cell: Cell, rows: int) -> tuple[frozenset[str], ...]:
        """The next `rows` lines of `cell`, taken off its queue.

        Raises:
            EvaluationError: If the queue holds fewer, which a request's own lines never allow:
                a cell is short by no more records than its lines not consumed.
        """
        queue = self._queues.get(cell)
        held = 0 if queue is None else len(queue)
        if queue is None or held < rows:
            msg = f"cell {cell} holds {held} lines for {rows} records asked"
            raise EvaluationError(msg)
        return tuple(queue.popleft() for _ in range(rows))

    def give_back(self, cell: Cell, lines: Sequence[frozenset[str]]) -> None:
        """Put `lines`, asked of `cell` and not consumed, at the end of its queue, in order."""
        self._queues.setdefault(cell, deque()).extend(lines)
        self.requeued += len(lines)
