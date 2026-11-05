"""Distance to closest record: how near rows sit to the training rows, by SDMetrics' distance.

The distance averages, over the columns, the range-normalised gap of a measured predictor and the
mismatch of a coded one; each row keeps its nearest training row. The privacy criterion reads the
5th percentile (ADR `research/e1-evaluation-protocol`). `nearest` computes SDMetrics'
`calculate_dcr` in blocks of plain arrays rather than through its pandas cross merge, and returns
its values bit for bit: each column's gap is `dcr_utils._process_dcr_chunk`'s, and the gaps are
summed one column at a time in frame column order, as the merged frame's row sum adds them, before
dividing by the column count, and a missing distance is skipped where rows meet, as its pandas
minimum skips it. `tests/test_evaluate_privacy.py` holds the two equal, missing values and a wholly
missing training column included. The lock holds SDMetrics' version.

`nearest_two` also keeps each row's second-nearest distance, which `nndr` reads, and
`nearest_from` reads each measured column's span from another frame than the one it measures
against.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.evaluate.scores import Measure


QUANTILE: Final = 0.05
"""The lower quantile of the distances the privacy criterion reads."""

BLOCK: Final = 2048
"""Rows and training rows per block; a block's gaps take `BLOCK` squared floats per array."""


def distances(rows: pd.DataFrame, train: pd.DataFrame, *, frame: FrameSpec) -> dict[Measure, float]:
    """The 5th percentile and the median of each row's distance to its closest training row."""
    nearest_distances = pd.Series(nearest(rows, train, frame=frame))
    return {
        Measure.DCR_P5: float(nearest_distances.quantile(QUANTILE)),
        Measure.DCR_MEDIAN: float(nearest_distances.median()),
    }


def nearest(
    rows: pd.DataFrame, train: pd.DataFrame, *, frame: FrameSpec, block: int = BLOCK
) -> np.ndarray:
    """Each row's distance to its closest training row, as SDMetrics' `calculate_dcr` gives it.

    Args:
        rows: The rows to measure, in `frame`'s columns.
        train: The training rows, in `frame`'s columns.
        frame: The frame whose numeric columns are measured and whose others are coded.
        block: Rows and training rows per block; the result does not depend on it.

    Returns:
        One distance per row of `rows`, in its order.

    Raises:
        ValueError: If `block` is below 1, or `train` holds no row, which leaves no distance.
    """
    return nearest_from(rows, train, frame=frame, spans_from=train, block=block)


def nearest_from(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    frame: FrameSpec,
    spans_from: pd.DataFrame,
    block: int = BLOCK,
) -> np.ndarray:
    """`nearest` from `rows` to `reference`, each measured column's span read from `spans_from`.

    The membership attack measures a record's distance to an arm's added rows on the training
    split's spans, so every arm's distances share one metric (`bko.evaluate.membership`).

    Raises:
        ValueError: As `nearest`, for `reference`.
    """
    result = np.empty(len(rows))
    for start, stop, block_distances in _blocks(rows, reference, frame, spans_from, block):
        best = np.full(stop - start, np.nan)
        for found in block_distances:
            np.fmin(best, np.fmin.reduce(found, axis=1), out=best)
        result[start:stop] = best
    return result


def nearest_two(
    rows: pd.DataFrame, train: pd.DataFrame, *, frame: FrameSpec, block: int = BLOCK
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's distances to its closest and its second-closest training rows.

    Each block keeps a running pair of minima, so the first equals `nearest` bit for bit. A
    missing distance is skipped, as `nearest` skips it; a row with no distance reads NaN.

    Raises:
        ValueError: If `block` is below 1, or `train` holds fewer than two rows.
    """
    if len(train) < 2:
        raise ValueError(f"a second-nearest distance needs two training rows, not {len(train)}")
    first, second = np.empty(len(rows)), np.empty(len(rows))
    for start, stop, block_distances in _blocks(rows, train, frame, train, block):
        low, next_low = np.full(stop - start, np.inf), np.full(stop - start, np.inf)
        for found in block_distances:
            found = np.where(np.isnan(found), np.inf, found)
            if found.shape[1] > 1:
                pair = np.partition(found, 1, axis=1)
                least, runner_up = pair[:, 0], pair[:, 1]
            else:
                least, runner_up = found[:, 0], np.full(stop - start, np.inf)
            next_low = np.minimum(np.maximum(low, least), np.minimum(next_low, runner_up))
            low = np.minimum(low, least)
        first[start:stop], second[start:stop] = low, next_low
    first[np.isinf(first)], second[np.isinf(second)] = np.nan, np.nan
    return first, second


def nndr(rows: pd.DataFrame, train: pd.DataFrame, *, frame: FrameSpec) -> dict[Measure, float]:
    """The nearest-neighbour distance ratio d1/d2 of each row to the training rows, summarised.

    0/0, a row with two training rows at distance 0, reads 1; d1 = 0 < d2, a copy of one
    training row, reads 0. It returns the ratios' 5th percentile and median, and how many rows
    read 0/0 (`NNDR_TIES`) and d1 = 0 < d2 (`NNDR_EXACT`); with no rows, each is NaN.

    Raises:
        ValueError: As `nearest_two`.
    """
    first, second = nearest_two(rows, train, frame=frame)
    if not len(rows):
        return dict.fromkeys(NNDR_MEASURES, math.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = pd.Series(np.where(second == 0, 1.0, first / second))
    return {
        Measure.NNDR_P5: float(ratios.quantile(QUANTILE)),
        Measure.NNDR_MEDIAN: float(ratios.median()),
        Measure.NNDR_TIES: float((second == 0).sum()),
        Measure.NNDR_EXACT: float(((first == 0) & (second > 0)).sum()),
    }


NNDR_MEASURES: Final = (
    Measure.NNDR_P5,
    Measure.NNDR_MEDIAN,
    Measure.NNDR_TIES,
    Measure.NNDR_EXACT,
)
"""What `nndr` returns, in order."""


def _blocks(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    frame: FrameSpec,
    spans_from: pd.DataFrame,
    block: int,
) -> Iterator[tuple[int, int, Iterator[np.ndarray]]]:
    """Each block of rows' start and stop, with its distances to each block of `reference`.

    Each column's gaps are summed one column at a time in frame column order before dividing by
    the column count, as SDMetrics' merged frame's row sum adds them.

    Raises:
        ValueError: If `block` is below 1, or `reference` holds no row.
    """
    if block < 1:
        raise ValueError(f"block must be at least 1, not {block}")
    if not len(reference):
        raise ValueError("no training rows to measure against, so no row has a distance")
    columns = [
        _column(
            rows[name], reference[name], numeric=name in frame.numeric, span_from=spans_from[name]
        )
        for name in frame.columns
    ]

    def distances_of(start: int, stop: int) -> Iterator[np.ndarray]:
        for at in range(0, len(reference), block):
            end = min(at + block, len(reference))
            total = columns[0].gap(start, stop, at, end)
            for column in columns[1:]:
                total += column.gap(start, stop, at, end)
            yield total / len(columns)

    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        yield start, stop, distances_of(start, stop)


class _Numeric:
    """A measured column: the absolute gap over the training range, capped at 1.

    The range is `span_from`'s: the reference's own for `nearest`, the training split's for
    `nearest_from`.
    """

    def __init__(self, data: pd.Series, reference: pd.Series, span_from: pd.Series) -> None:
        self.data: np.ndarray = data.to_numpy(dtype=float, na_value=np.nan)
        self.reference: np.ndarray = reference.to_numpy(dtype=float, na_value=np.nan)
        spanned = span_from.to_numpy(dtype=float, na_value=np.nan)
        present = spanned[~np.isnan(spanned)]
        self.span: float = float(present.max() - present.min()) if present.size else math.nan

    def gap(self, start: int, stop: int, reference: int, end: int) -> np.ndarray:
        data = self.data[start:stop, None]
        ref = self.reference[None, reference:end]
        diff = np.abs(ref - data)
        gap = (diff > 0).astype(float) if self.span == 0 else np.minimum(diff / self.span, 1.0)
        data_missing, ref_missing = np.isnan(data), np.isnan(ref)
        gap[data_missing ^ ref_missing] = 1.0
        gap[data_missing & ref_missing] = 0.0
        return gap


class _Coded:
    """A coded column: 1 where the codes differ, 0 where they match or both are missing."""

    def __init__(self, data: pd.Series, reference: pd.Series) -> None:
        codes, _ = pd.factorize(pd.concat([data, reference], ignore_index=True))
        self.data: np.ndarray = codes[: len(data)]
        self.reference: np.ndarray = codes[len(data) :]

    def gap(self, start: int, stop: int, reference: int, end: int) -> np.ndarray:
        data = self.data[start:stop, None]
        ref = self.reference[None, reference:end]
        return (data != ref).astype(float)


def _column(
    data: pd.Series, reference: pd.Series, *, numeric: bool, span_from: pd.Series | None = None
) -> _Numeric | _Coded:
    """A column's gaps; a measured column's span is `span_from`'s, `reference`'s when None."""
    if not numeric:
        return _Coded(data, reference)
    return _Numeric(data, reference, reference if span_from is None else span_from)
