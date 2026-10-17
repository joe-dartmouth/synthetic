"""Fidelity of synthetic rows to reference rows: SDMetrics' quality report over every column pair.

The report's default thresholds leave out every column pair whose real correlation or association
is weak, which on this frame drops most pairs and changes the set from split to split; both are set
to 0, so the same pairs count every time.
"""

from __future__ import annotations

import copy
from typing import Final

import pandas as pd
from sdmetrics.reports import QualityReport

from bko.baselines.synthesizers import METADATA
from bko.datasets.uci import FRAME_COLUMNS
from bko.evaluate.scores import Measure


TABLE: Final = "uci"
"""The table `METADATA` describes."""


def metadata() -> dict[str, object]:
    """`METADATA` as the plain, private dict SDMetrics requires."""
    return copy.deepcopy(dict(METADATA))


def quality_report(reference: pd.DataFrame, synthetic: pd.DataFrame) -> QualityReport:
    """SDMetrics' quality report of `synthetic` against `reference`, generated quietly."""
    columns = list(FRAME_COLUMNS)
    report = QualityReport()
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0
    report.generate(
        {TABLE: reference.loc[:, columns]},
        {TABLE: synthetic.loc[:, columns]},
        metadata(),
        verbose=False,
    )
    return report


def fidelity(reference: pd.DataFrame, synthetic: pd.DataFrame) -> dict[Measure, float]:
    """The quality score and its two properties for `synthetic` against `reference`."""
    report = quality_report(reference, synthetic)
    properties = report.get_properties().set_index("Property")["Score"]
    return {
        Measure.QUALITY: float(report.get_score()),
        Measure.COLUMN_SHAPES: float(properties["Column Shapes"]),
        Measure.COLUMN_PAIR_TRENDS: float(properties["Column Pair Trends"]),
    }
