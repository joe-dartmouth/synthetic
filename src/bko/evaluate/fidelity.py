"""Fidelity of synthetic rows to reference rows: SDMetrics' quality report over every column pair.

The report's default thresholds leave out every column pair whose real correlation or association
is weak, which on this frame drops most pairs and changes the set from split to split; both are set
to 0, so the same pairs count every time. The table is named for the frame, which each function
requires.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    frame: FrameSpec,
    columns: Sequence[str] | None = None,
) -> QualityReport:
    """SDMetrics' quality report of `synthetic` against `reference`, generated quietly.

    It reads `columns` of `frame`, or every column when None. SDMetrics subsamples each table to
    50,000 rows for its contingency column pairs, with no seed, when either is larger, and its
    single-table report does not pass its own `num_rows_subsample` down to the per-table property
    that draws the sample (sdmetrics 0.32). `_every_row` sets that property to read every row, so a
    reading of a larger table replays exactly.
    """
    chosen = list(frame.columns if columns is None else columns)
    described = metadata(frame)
    table: Any = described["tables"]
    table[frame.name]["columns"] = {name: table[frame.name]["columns"][name] for name in chosen}
    report = QualityReport()
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0
    _every_row(report)
    report.generate(
        {frame.name: reference.loc[:, chosen]},
        {frame.name: synthetic.loc[:, chosen]},
        described,
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


@dataclass(frozen=True, slots=True)
class ReportScores:
    """A quality report's two properties and the column shapes it read.

    Attributes:
        shapes: The column shapes score.
        pair_trends: The column pair trends score.
        columns: Each requested column's shape, in the requested order.
    """

    shapes: float
    pair_trends: float
    columns: Mapping[str, float]


def report_scores(report: QualityReport, *, columns: Sequence[str]) -> ReportScores:
    """`report`'s column shapes and pair trends scores, and each of `columns`' shape.

    A property the report's properties lack, or a column its column-shapes details lack, reads NaN.
    """
    properties = report.get_properties().set_index("Property")["Score"]
    details = report.get_details("Column Shapes").set_index("Column")["Score"]
    return ReportScores(
        float(properties.get("Column Shapes", math.nan)),
        float(properties.get("Column Pair Trends", math.nan)),
        {name: float(details.get(name, math.nan)) for name in columns},
    )


def fidelity(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, frame: FrameSpec
) -> dict[Measure, float]:
    """The quality score and its two properties for `synthetic` against `reference`."""
    report = quality_report(reference, synthetic, frame=frame)
    found = report_scores(report, columns=())
    return {
        Measure.QUALITY: float(report.get_score()),
        Measure.COLUMN_SHAPES: found.shapes,
        Measure.COLUMN_PAIR_TRENDS: found.pair_trends,
    }
