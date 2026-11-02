"""Distance to closest record: how near rows sit to the training rows, by SDMetrics' distance.

The distance averages, over the columns, the range-normalised gap of a measured predictor and the
mismatch of a coded one; each row keeps its nearest training row. The privacy criterion reads the
5th percentile (ADR `research/e1-evaluation-protocol`). `nearest` computes SDMetrics'
`calculate_dcr` in blocks of plain arrays rather than through its pandas cross merge, and returns
its values bit for bit: each column's gap is `dcr_utils._process_dcr_chunk`'s, and the gaps are
summed one column at a time in frame column order, as the merged frame's row sum adds them, before
dividing by the column count. `tests/test_evaluate_privacy.py` holds the two equal. The lock holds
SDMetrics' version.
"""

from __future__ import annotations

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
        ValueError: If `block` is below 1.
    """
    if block < 1:
        raise ValueError(f"block must be at least 1, not {block}")
    columns = [
        _column(rows[name], train[name], numeric=name in frame.numeric) for name in frame.columns
    ]
    result = np.empty(len(rows))
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        best = np.full(stop - start, np.inf)
        for reference in range(0, len(train), block):
            end = min(reference + block, len(train))
            total = columns[0].gap(start, stop, reference, end)
            for column in columns[1:]:
                total += column.gap(start, stop, reference, end)
            np.minimum(best, (total / len(columns)).min(axis=1), out=best)
        result[start:stop] = best
    return result


class _Numeric:
    """A measured column: the absolute gap over the training range, capped at 1."""

    def __init__(self, data: pd.Series, reference: pd.Series) -> None:
        self.data: np.ndarray = data.to_numpy(dtype=float, na_value=np.nan)
        self.reference: np.ndarray = reference.to_numpy(dtype=float, na_value=np.nan)
        self.span: float = float(reference.max() - reference.min())

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


def _column(data: pd.Series, reference: pd.Series, *, numeric: bool) -> _Numeric | _Coded:
    return _Numeric(data, reference) if numeric else _Coded(data, reference)
