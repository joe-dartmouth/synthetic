"""Validated records joined to the training split's records.

ADR `research/gap-closure-design` states which records join; `join` applies it and counts what it
drops. The joined rows are indexed -1, -2, ... under the training frame's index name, as
`bko.baselines.records.number` numbers added rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from bko.datasets.cells import cell_keys
from bko.gaps.observe import Cell, cell_key


@dataclass(frozen=True, slots=True, eq=False)
class Joined:
    """The training records with the validated records that joined them.

    Attributes:
        frame: The training records, then each joined record in request order.
        joined: How many records joined each requested cell, in request order.
        off_cell: Accepted records dropped per requested cell for naming another cell.
        past_quota: Accepted on-cell records dropped per requested cell past its rows.
    """

    frame: pd.DataFrame
    joined: Mapping[Cell, int]
    off_cell: Mapping[Cell, int]
    past_quota: Mapping[Cell, int]


def join(
    train: pd.DataFrame,
    accepted: Mapping[Cell, pd.DataFrame],
    request: Mapping[Cell, int],
    *,
    edges: Sequence[int],
) -> Joined:
    """Join each requested cell's accepted records to `train`, in request order.

    Args:
        train: The training split.
        accepted: Each requested cell's `Validated.accepted`, in generation order; a requested
            cell it lacks joins nothing.
        request: Rows per requested cell.
        edges: Age band starts.

    Raises:
        ValueError: If `accepted` names a cell `request` does not ask for, or a frame's columns
            or dtypes differ from `train`'s.
    """
    unasked = [cell_key(cell) for cell in accepted if cell not in request]
    if unasked:
        msg = f"accepted records for cells the request does not ask for: {', '.join(unasked)}"
        raise ValueError(msg)
    kept: list[pd.DataFrame] = [train.iloc[:0]]
    joined: dict[Cell, int] = {}
    off_cell: dict[Cell, int] = {}
    past_quota: dict[Cell, int] = {}
    for cell, rows in request.items():
        frame = accepted.get(cell, train.iloc[:0])
        if list(frame.columns) != list(train.columns) or not frame.dtypes.equals(train.dtypes):
            msg = (
                f"{cell_key(cell)}'s accepted records differ from the training split's "
                "columns or dtypes"
            )
            raise ValueError(msg)
        keys = cell_keys(frame, edges=edges)
        mask = pd.Series(True, index=frame.index)
        for column, part in zip(keys.columns, cell, strict=True):
            mask &= keys[column] == part
        on_cell = frame.loc[mask]
        kept.append(on_cell.iloc[:rows])
        joined[cell] = min(len(on_cell), rows)
        off_cell[cell] = len(frame) - len(on_cell)
        past_quota[cell] = max(0, len(on_cell) - rows)
    added = pd.concat(kept)
    added.index = pd.RangeIndex(-1, -len(added) - 1, -1, name=train.index.name)
    return Joined(
        frame=pd.concat([train, added], verify_integrity=True),
        joined=MappingProxyType(joined),
        off_cell=MappingProxyType(off_cell),
        past_quota=MappingProxyType(past_quota),
    )
