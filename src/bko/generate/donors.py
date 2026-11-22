"""The training records a direct draw copies field values from: label groups and donor picks.

`label_group` observes a label's training records as the group whose prior a share reads, and
`_pick` draws a donor record from a cell's matches, else from its label group's (ADRs
`research/bkb-direct-generation` and `research/data-contradiction-handling`).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, Observed, cell_key, observe_records
from bko.generate.errors import SamplingError
from bko.ontology.schema import FieldMapping


type _Source = tuple[int, dict[str, float]]
"""A training record with its id."""


@dataclass(frozen=True, slots=True, eq=False)
class _Group:
    """A label's training records: observed, as records, and each record's age band."""

    observed: Observed
    records: list[_Source]
    bands: list[str]


def _pick(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
) -> tuple[_Source, bool]:
    """A record drawn from `own`, else from `other`, with its id, and whether it fell back.

    Raises:
        SamplingError: If both are empty.
    """
    if own:
        return own[int(rng.integers(len(own)))], False
    if other:
        return other[int(rng.integers(len(other)))], True
    msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no training record matches the draw"
    raise SamplingError(msg)


def label_group(
    train: pd.DataFrame, label: int, rows: Sequence[FieldMapping], *, name: str, frame: FrameSpec
) -> Observed:
    """The training records of `label`, observed as the label group whose prior a share reads.

    Raises:
        GapError: As `observe_records`.
    """
    members = train.loc[train[frame.label].eq(label)]
    return _observe_group(members, label, rows, name=name, frame=frame)


def _observe_group(
    members: pd.DataFrame,
    label: int,
    rows: Sequence[FieldMapping],
    *,
    name: str,
    frame: FrameSpec,
) -> Observed:
    """`members`, the training records of `label`, observed under the label group's name."""
    return observe_records(members, rows, name=f"{name}/{frame.label}={label}")


def _label_groups(
    train: pd.DataFrame,
    keys: pd.DataFrame,
    labels: Collection[int],
    rows: Sequence[FieldMapping],
    name: str,
    frame: FrameSpec,
) -> dict[int, _Group]:
    groups: dict[int, _Group] = {}
    for label in sorted(labels):
        members = train.loc[train[frame.label].eq(label)]
        observed = _observe_group(members, label, rows, name=name, frame=frame)
        bands = [str(band) for band in keys.loc[members.index, keys.columns[0]]]
        groups[label] = _Group(observed, _records(members, frame), bands)
    return groups


def _records(frame: pd.DataFrame, spec: FrameSpec) -> list[_Source]:
    """Each record of `frame` with its id, in row order."""
    columns = [str(column) for column in spec.columns]
    return [
        (int(index), dict(zip(columns, values, strict=True)))
        for index, *values in frame[columns].itertuples(name=None)
    ]
