"""Fidelity of synthetic rows to reference rows: SDMetrics' quality report over every column pair.

The report's default thresholds leave out every column pair whose real correlation or association
is weak, which on this frame drops most pairs and changes the set from split to split; both are set
to 0, so the same pairs count every time. The table is named for the frame, which each function
requires.
"""

from __future__ import annotations

import copy
from typing import Final

import pandas as pd
from sdmetrics.reports import QualityReport

from bko.baselines import synthesizers
from bko.datasets.frames import FrameSpec
from bko.evaluate.scores import Measure


MEASURES: Final = (Measure.QUALITY, Measure.COLUMN_SHAPES, Measure.COLUMN_PAIR_TRENDS)
"""The measures `fidelity` returns, in its order."""


def metadata(frame: FrameSpec) -> dict[str, object]:
    """`frame`'s SDV metadata as the plain, private dict SDMetrics requires."""
    return copy.deepcopy(dict(synthesizers.metadata(frame)))


def quality_report(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, frame: FrameSpec
) -> QualityReport:
    """SDMetrics' quality report of `synthetic` against `reference`, generated quietly."""
    columns = list(frame.columns)
    report = QualityReport()
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0
    report.generate(
        {frame.name: reference.loc[:, columns]},
        {frame.name: synthetic.loc[:, columns]},
        metadata(frame),
        verbose=False,
    )
    return report


def fidelity(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, frame: FrameSpec
) -> dict[Measure, float]:
    """The quality score and its two properties for `synthetic` against `reference`."""
    report = quality_report(reference, synthetic, frame=frame)
    properties = report.get_properties().set_index("Property")["Score"]
    return {
        Measure.QUALITY: float(report.get_score()),
        Measure.COLUMN_SHAPES: float(properties["Column Shapes"]),
        Measure.COLUMN_PAIR_TRENDS: float(properties["Column Pair Trends"]),
    }
