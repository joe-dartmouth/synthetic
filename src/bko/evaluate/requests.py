"""The requests an `Experiment` gives every arm: the gap fill and the replica of the cells.

`bko.evaluate.experiment` re-exports the three public builders, so every importer reads them there.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

import pandas as pd

from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.split import dropped_count, kept_count
from bko.evaluate.errors import EvaluationError


if TYPE_CHECKING:
    from bko.evaluate.experiment import Experiment


def flagged(
    train: pd.DataFrame, experiment: Experiment, *, edges: Sequence[int]
) -> tuple[Stratum, ...]:
    """The cells under `edges` holding fewer than `experiment.threshold` records, in grid order.

    With `experiment.flagged_cell` set, that cell alone, whatever its count.
    """
    if experiment.flagged_cell is not None:
        return (experiment.flagged_cell,)
    return tuple(_short(train, experiment, edges=edges))


def gap_request(
    train: pd.DataFrame, experiment: Experiment, *, seed: int, frame: FrameSpec
) -> Request:
    """Ask for every flagged cell of `frame` to be filled up to `experiment.fill_to` records.

    A cell is flagged below `experiment.threshold`, so a target above it fills the same cells
    further. With `experiment.flagged_cell` set, the request asks for that cell alone, up to
    `fill_to`. With `experiment.synthetic_share` set, `train` is the whole training split and the
    request asks each of its cells for the rows the share drops there (`dropped_count`), leaving
    out each cell that drops none; no cell is restored to `fill_to`.

    Raises:
        EvaluationError: If no cell is flagged, so the row-adding arms have nothing to fill, or
            the flagged cell is not a cell of `frame` or already holds `fill_to` records, or the
            synthetic share drops no row of any cell.
    """
    share = experiment.synthetic_share
    if share is not None:
        cells = _cells(train, edges=frame.edges)
        quota = {cell: asked for cell, count, _ in cells if (asked := dropped_count(count, share))}
        if not quota:
            msg = f"the synthetic share {share} drops no training row; nothing to fill"
            raise EvaluationError(msg)
    elif experiment.flagged_cell is not None:
        quota = _restored(train, experiment, experiment.flagged_cell, frame=frame)
    else:
        quota = _short(train, experiment, edges=frame.edges)
        if not quota:
            msg = f"no cell holds fewer than {experiment.threshold} training records;"
            msg += " nothing to fill"
            raise EvaluationError(msg)
    return Request(quota=quota, strata=Strata.CELL, seed=seed, frame=frame)


def replica_request(
    train: pd.DataFrame, experiment: Experiment, *, seed: int, frame: FrameSpec
) -> Request:
    """Ask for as many rows per cell as the training split holds: a synthetic copy of its cells.

    With `experiment.replica_rows` set, each cell asks for its `kept_count` at the share
    `replica_rows / len(train)`, so the cap is split across cells in proportion and the rounded
    cells may sum to a few rows off it; a cap above the training rows asks more rows than they
    hold, at the same share.
    """
    quota = {cell: count for cell, count, _ in _cells(train, edges=frame.edges) if count}
    if experiment.replica_rows is not None:
        share = Fraction(experiment.replica_rows, len(train))
        quota = {cell: kept_count(count, share) for cell, count in quota.items()}
    return Request(quota=quota, strata=Strata.CELL, seed=seed, frame=frame)


def _restored(
    train: pd.DataFrame, experiment: Experiment, cell: Stratum, *, frame: FrameSpec
) -> dict[Stratum, int]:
    counts = {found: count for found, count, _ in _cells(train, edges=frame.edges)}
    if cell not in counts:
        msg = f"the flagged cell {cell} is not a cell of the frame {frame.name}"
        raise EvaluationError(msg)
    asked = experiment.fill_to - counts[cell]
    if asked <= 0:
        msg = (
            f"the flagged cell {cell} holds {counts[cell]} training records, so fill_to "
            f"{experiment.fill_to} adds none"
        )
        raise EvaluationError(msg)
    return {cell: asked}


def _short(
    train: pd.DataFrame, experiment: Experiment, *, edges: Sequence[int]
) -> dict[Stratum, int]:
    return {
        cell: experiment.fill_to - count
        for cell, count, short in _cells(train, experiment.threshold, edges=edges)
        if short
    }


def _cells(
    train: pd.DataFrame,
    below: int = UNDERREPRESENTED_BELOW,
    *,
    edges: Sequence[int],
) -> list[tuple[Stratum, int, bool]]:
    counts = cell_counts(train, edges=edges, below=below)
    return [
        ((str(band), str(sex), int(label)), int(count), bool(short))
        for band, sex, label, count, short in counts.itertuples(index=False)
    ]
