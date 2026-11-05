"""Fidelity of synthetic rows to reference rows: SDMetrics' quality report over every column pair.

The report's default thresholds leave out every column pair whose real correlation or association
is weak, which on this frame drops most pairs and changes the set from split to split; both are set
to 0, so the same pairs count every time. The table is named for the frame, which each function
requires.
"""

from __future__ import annotations

import copy
from typing import Any, Final

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
    """SDMetrics' quality report of `synthetic` against `reference`, generated quietly.

    SDMetrics subsamples each table to 50,000 rows for its contingency column pairs, with no seed,
    when either is larger, and its single-table report does not pass its own `num_rows_subsample`
    down to the per-table property that draws the sample (sdmetrics 0.32). `_every_row` sets that
    property to read every row, so a reading of a larger table replays exactly.
    """
    columns = list(frame.columns)
    report = QualityReport()
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0
    _every_row(report)
    report.generate(
        {frame.name: reference.loc[:, columns]},
        {frame.name: synthetic.loc[:, columns]},
        metadata(frame),
        verbose=False,
    )
    return report


def _every_row(report: QualityReport) -> None:
    """Make `report`'s column pair trends read every row of each table, never a subsample."""
    # Typed as Any: the hook is a method SDMetrics' stubs do not declare assignable.
    pairs: Any = report._properties["Column Pair Trends"]
    configure = pairs._configure_single_table_property

    def every_row(table_name: str) -> None:
        configure(table_name)
        pairs._properties[table_name].num_rows_subsample = None

    pairs._configure_single_table_property = every_row


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
