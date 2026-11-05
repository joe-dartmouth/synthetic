"""The BKB likelihood scorer G7's gate and the ranking arm `bko-curated` share.

`scorer` builds one cell's `bko.gaps.likelihood.Likelihood` from a holdout's training rows: the
cell's shares (`bko.gaps.shares.shares`) read the cell's own records and its label group's.
`tie` is the order that breaks equal scores, `sha256(f"{holdout}:{call}:{position}")`, so neither
the gate's split nor the arm's ranking favours an early call (ADR
`research/bkb-candidate-curation`).
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import pandas as pd

from bko.baselines.request import Strata, stratum_keys
from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError
from bko.gaps.likelihood import Likelihood
from bko.gaps.observe import Cell, cell_key, observe_records, record_cells
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


def scorer(train: pd.DataFrame, data: Known, frame: FrameSpec, cell: Cell) -> Likelihood:
    """The likelihood of a record of `cell` at the cell's shares on `train`.

    Args:
        train: The holdout's training rows, in `frame`'s columns.
        data: The run's source name and field mappings.
        frame: The frame whose edges cut the cells and whose label marks the label-only concepts.
        cell: The cell, as `bko.gaps.observe.record_cells` keys it.

    Raises:
        EvaluationError: If `train` holds no record of `cell`.
    """
    rows = data.field_rows
    keys = stratum_keys(train, Strata.CELL, edges=frame.edges)
    members = train.loc[[key == cell for key in keys]]
    if members.empty:
        msg = f"the training rows hold no record of cell {cell_key(cell)!r}"
        raise EvaluationError(msg)
    found = record_cells(members, rows, name=data.name, edges=frame.edges).each[cell]
    label = int(cell[2])
    group = observe_records(
        train.loc[train[frame.label].eq(label)], rows, name=f"{data.name}/{frame.label}={label}"
    )
    return Likelihood(shares(found, group), rows, frame)


def tie(holdout: int, call: int | None, position: int) -> str:
    """The order equal scores break by: `sha256(f"{holdout}:{call}:{position}")`, hex."""
    return hashlib.sha256(f"{holdout}:{call}:{position}".encode()).hexdigest()
