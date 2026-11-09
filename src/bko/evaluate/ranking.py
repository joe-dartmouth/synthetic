"""The BKB likelihood scorer G7's gate and the ranking arm `bko-curated` share.

`Scorers` builds each cell's `bko.gaps.likelihood.Likelihood` from one holdout's training rows:
the cell's shares (`bko.gaps.shares.shares`) read the cell's own records and its label group's,
and the holdout's stratum keys and each label group's observations are computed once.
`record` reads a valid candidate as the scorer takes it, and `tie` is the order that breaks equal
scores, `sha256(f"{holdout}:{call}:{position}")`, so neither the gate's split nor the arm's
ranking favours an early call (ADR `research/bkb-candidate-curation`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from bko.baselines.request import Strata, stratum_keys
from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError
from bko.gaps.likelihood import Likelihood
from bko.gaps.observe import Cell, Observed, cell_key, observe_records, record_cells
from bko.gaps.shares import shares
from bko.ontology.schema import FieldMapping


class Known(Protocol):
    """What a scorer reads of the run's data: its source name and its field mappings."""

    @property
    def name(self) -> str:
        """The dataset's source name; each cell's fragment takes it."""
        ...

    @property
    def field_rows(self) -> tuple[FieldMapping, ...]:
        """The field mappings the concepts are read by."""
        ...


class Scorers:
    """Each cell's likelihood on one holdout's training rows, sharing the holdout's work.

    The stratum keys are computed once, and each label group is observed once, on its first
    cell; each `Likelihood` is built from the same inputs a per-cell build reads.
    """

    def __init__(self, train: pd.DataFrame, data: Known, frame: FrameSpec) -> None:
        """Hold `train`'s stratum keys under `frame`'s edges; observe no group yet.

        Args:
            train: The holdout's training rows, in `frame`'s columns.
            data: The run's source name and field mappings.
            frame: The frame whose edges cut the cells and whose label marks the label-only
                concepts.
        """
        self._train = train
        self._data = data
        self._frame = frame
        self._keys = stratum_keys(train, Strata.CELL, edges=frame.edges)
        self._groups: dict[int, Observed] = {}

    def __call__(self, cell: Cell) -> Likelihood:
        """The likelihood of a record of `cell` at the cell's shares.

        Args:
            cell: The cell, as `bko.gaps.observe.record_cells` keys it.

        Returns:
            A `Likelihood` that reads a record's log-probability at the cell's shares, minus
            infinity for a record the sampler cannot draw.

        Raises:
            EvaluationError: If the training rows hold no record of `cell`.
        """
        rows = self._data.field_rows
        members = self._train.loc[[key == cell for key in self._keys]]
        if members.empty:
            msg = f"the training rows hold no record of cell {cell_key(cell)!r}"
            raise EvaluationError(msg)
        edges = self._frame.edges
        found = record_cells(members, rows, name=self._data.name, edges=edges).each[cell]
        label = int(cell[2])
        if label not in self._groups:
            group = self._train.loc[self._train[self._frame.label].eq(label)]
            name = f"{self._data.name}/{self._frame.label}={label}"
            self._groups[label] = observe_records(group, rows, name=name)
        return Likelihood(shares(found, self._groups[label]), rows, self._frame)


def over_ask_error(over_ask: int, *, ranks: bool) -> str | None:
    """Why `over_ask` is invalid for an arm that ranks its candidates or not; None if valid.

    `over_ask` is 1 or more, and above 1 exactly when the arm ranks.

    Args:
        over_ask: The multiple of each cell's quota the arm asks for.
        ranks: Whether the arm ranks its candidates (`Curation.LIKELIHOOD`, or an attempt's
            `rank`).

    Returns:
        The refusal's message, its specific reason first, then the rule; None for a valid
        `over_ask`.
    """
    if over_ask < 1:
        reason = "is below 1"
    elif over_ask > 1 and not ranks:
        reason = "is above 1 but the arm does not rank"
    elif over_ask == 1 and ranks:
        reason = "is 1 but the arm ranks, so it has nothing to drop"
    else:
        return None
    return (
        f"over_ask {over_ask} {reason}; over_ask is 1 or more, and above 1 exactly when the arm "
        "ranks"
    )


def tie(holdout: int, call: int | None, position: int) -> str:
    """The order equal scores break by: `sha256(f"{holdout}:{call}:{position}")`, hex.

    Args:
        holdout: The holdout's seed.
        call: The candidate's call index, or None for a record drawn with no call.
        position: The candidate's position in its call's reply.

    Returns:
        The digest as hex text; equal scores keep the lower digest first.
    """
    return hashlib.sha256(f"{holdout}:{call}:{position}".encode()).hexdigest()


def record(candidate: object, columns: Sequence[str]) -> dict[str, float]:
    """A valid candidate's values in `columns`, as the scorer reads them.

    Args:
        candidate: A candidate that passed validation, as its `Seen` holds it.
        columns: The frame's columns, in order.

    Returns:
        Each of `columns` with the candidate's value as a float, in `columns`' order.

    Raises:
        EvaluationError: If `candidate` is not a record holding every column.
    """
    if not isinstance(candidate, Mapping) or not set(columns) <= set(candidate):
        msg = f"a ranked candidate is a record of every column; got {candidate!r}"
        raise EvaluationError(msg)
    return {column: float(candidate[column]) for column in columns}
