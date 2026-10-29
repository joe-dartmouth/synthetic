"""Distance to closest record: how near rows sit to the training rows, by SDMetrics' distance.

`calculate_dcr` averages, over the columns, the range-normalised gap of a measured predictor and
the mismatch of a coded one, then keeps the nearest training row. The privacy criterion reads the
5th percentile (ADR `research/e1-evaluation-protocol`). `calculate_dcr` lives in a module SDMetrics
does not export; the lock holds its version.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr

from bko.datasets.frames import FrameSpec
from bko.evaluate.fidelity import metadata
from bko.evaluate.scores import Measure


QUANTILE: Final = 0.05
"""The lower quantile of the distances the privacy criterion reads."""


def distances(rows: pd.DataFrame, train: pd.DataFrame, *, frame: FrameSpec) -> dict[Measure, float]:
    """The 5th percentile and the median of each row's distance to its closest training row."""
    columns = list(frame.columns)
    nearest = calculate_dcr(
        dataset=rows.loc[:, columns],
        reference_dataset=train.loc[:, columns],
        metadata=metadata(frame),
    )
    return {
        Measure.DCR_P5: float(nearest.quantile(QUANTILE)),
        Measure.DCR_MEDIAN: float(nearest.median()),
    }
