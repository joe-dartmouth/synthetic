"""Perturbing `bko-direct`'s unmapped fields: the fields, the reading and the shared arm.

`bko-direct` copies each field no field row reads from its own training record
(`bko.generate.direct`), so its rows are mosaics of training values. `unmapped` names those fields:
every column but the field rows' fields, `age`, `sex` and the label. On the Diabetes frame they
must be `DIABETES_UNMAPPED`, the 19 a rewrite call may send; `unmapped` refuses another set there.
`Perturbing` holds what both perturbation arms share; `bko.evaluate.pram.PramArm` redraws the
fields, and `bko.evaluate.perturbed.PerturbedArm` rewrites them through calls merged by
`bko.evaluate.perturbed.merge`.

`read_distribution` reads a holdout's perturbed rows and its unperturbed rows against the label-1
training rows over the perturbed fields: SDMetrics' column shapes, column pair trends and each
column's shape, and on the Diabetes frame the share of rows `incoherent` counts. The reading never
changes a row; each arm keeps its own rows on every holdout. ADR
`research/identifier-perturbation` records both arms and the reading.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from bko.baselines.fills import CURRENT_FILL
from bko.baselines.request import Request, Stratum
from bko.datasets.diabetes import CHANGE_CODES, DIABETES_MED_CODES, DRUG_CODES, DRUGS
from bko.datasets.frames import FrameSpec
from bko.evaluate.diabetes import DIABETES_NAME
from bko.evaluate.errors import EvaluationError
from bko.evaluate.fidelity import quality_report
from bko.evaluate.holdout_candidates import MOVED
from bko.evaluate.scores import Measure, Source
from bko.generate.direct import field_groups, free_fields
from bko.ontology.schema import FieldMapping


if TYPE_CHECKING:
    from bko.evaluate.attempts import Tally
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)

POSITIVE: Final = 1
"""The label whose training rows are the reading's reference."""

DIABETES_UNMAPPED: Final = (
    "race",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "glimepiride",
    "repaglinide",
    "change",
    "diabetesMed",
)
"""The 19 Diabetes fields no field row reads, in frame order: all a rewrite call may send."""

PINNED: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({DIABETES_NAME: DIABETES_UNMAPPED})
"""Each frame's pinned perturbed fields, by frame name; every named run that perturbs is on one."""

type Record = dict[str, int | float]
"""One drawn row's value per column."""


def unmapped(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> tuple[str, ...]:
    """`frame`'s columns that no field row reads, but `age`, `sex` and the label, in frame order.

    Raises:
        EvaluationError: If `frame` has `PINNED` fields and they are not these.
    """
    fields = free_fields(rows, frame=frame)
    pinned = PINNED.get(frame.name)
    if pinned is not None and fields != pinned:
        msg = (
            f"frame {frame.name}'s unmapped fields {list(fields)} are not its pinned {list(pinned)}"
        )
        raise EvaluationError(msg)
    return fields


@dataclass(frozen=True, slots=True)
class Changed:
    """One drawn row once a perturbation step has read it.

    Attributes:
        record: Its values.
        call: The call that returned it; None for a row no call returned.
        position: Its row index within that call; 0 with no call.
        result: What the step did, one of its arm's `bko.evaluate.holdout_candidates.PERTURBS`;
            None for a draw no step read. Every perturbation step sets it on every row, since
            `Perturbing.read` counts a row with None as unchanged.
    """

    record: Record
    call: int | None = None
    position: int = 0
    result: str | None = None


type Change = Callable[
    [pd.DataFrame, Request, Mapping[Stratum, list[Record]]], Mapping[Stratum, list[Changed]]
]
"""A perturbation step: from the training rows, the request and each cell's drawn records, in
quota order, each cell's rows in the same order (`bko.evaluate.direct.DirectArm.fill`)."""


@dataclass(frozen=True, slots=True)
class Quality:
    """SDMetrics' reading of some rows against a reference, over the fields read.

    Attributes:
        shapes: The column shapes score.
        pair_trends: The column pair trends score.
        columns: Each field's column shape, in the fields' order.
    """

    shapes: float
    pair_trends: float
    columns: Mapping[str, float]


def quality(
    rows: pd.DataFrame, reference: pd.DataFrame, *, fields: Sequence[str], frame: FrameSpec
) -> Quality:
    """SDMetrics' column shapes, pair trends and each column's shape of `rows` against `reference`.

    A field SDMetrics' column-shapes details lack reads NaN, with a `perturb unscored` WARNING.
    """
    report = quality_report(reference, rows, frame=frame, columns=fields)
    properties = report.get_properties().set_index("Property")["Score"]
    details = report.get_details("Column Shapes").set_index("Column")["Score"]
    for name in fields:
        if name not in details.index:
            logger.warning("perturb unscored holdout=%d field=%s", _holdout(), name)
    columns = {name: float(details.get(name, math.nan)) for name in fields}
    return Quality(
        float(properties["Column Shapes"]), float(properties["Column Pair Trends"]), columns
    )


def incoherent(rows: pd.DataFrame) -> float:
    """The share of Diabetes `rows` that break any of three rules every real training row keeps.

    The rules: a drug of `DRUGS` not `No` with `diabetesMed` `No`; `change` `Ch` with
    `diabetesMed` `No`; a drug `Up` or `Down` with `change` `No`. NaN for no rows.
    """
    drugs = rows[list(DRUGS)]
    unmedicated = rows["diabetesMed"] == DIABETES_MED_CODES["No"]
    prescribed = (drugs != DRUG_CODES["No"]).any(axis=1)
    dosed = drugs.isin([DRUG_CODES["Up"], DRUG_CODES["Down"]]).any(axis=1)
    broken = (
        (prescribed & unmedicated)
        | ((rows["change"] == CHANGE_CODES["Ch"]) & unmedicated)
        | (dosed & (rows["change"] == CHANGE_CODES["No"]))
    )
    return float(broken.mean()) if len(rows) else math.nan


@dataclass(frozen=True, slots=True)
class Reading:
    """One holdout's reading of an arm's perturbed rows beside its unperturbed rows.

    Attributes:
        perturbed: The perturbed rows' quality against the reference rows.
        unperturbed: The unperturbed rows' quality against them.
        incoherent: The perturbed rows' share breaking a Diabetes rule; None off Diabetes.
        unperturbed_incoherent: The unperturbed rows' share; None off Diabetes.
    """

    perturbed: Quality
    unperturbed: Quality
    incoherent: float | None
    unperturbed_incoherent: float | None

    @property
    def difference(self) -> float:
        """The perturbed rows' column shapes less the unperturbed rows'."""
        return self.perturbed.shapes - self.unperturbed.shapes


def read_distribution(
    perturbed: pd.DataFrame,
    original: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    fields: Sequence[str],
    frame: FrameSpec,
) -> Reading:
    """Read `perturbed` and `original` against `reference` over `fields`.

    `incoherent` is read only on the Diabetes frame.
    """
    diabetes = frame.name == DIABETES_NAME
    return Reading(
        quality(perturbed, reference, fields=fields, frame=frame),
        quality(original, reference, fields=fields, frame=frame),
        incoherent(perturbed) if diabetes else None,
        incoherent(original) if diabetes else None,
    )


def _share(value: float | None) -> str:
    """`value` at six places, or `none`."""
    return "none" if value is None else f"{value:.6f}"


def _columns(read: Quality) -> str:
    """Each field's column shape as `field=value`, in the fields' order."""
    return " ".join(f"{name}={value:.6f}" for name, value in read.columns.items())


def _mean(values: Sequence[float]) -> str:
    """The mean of `values` at six places; `none` for none."""
    return f"{math.fsum(values) / len(values):.6f}" if values else "none"


def drawn_frame(drawn: Mapping[Stratum, list[Record]], frame: FrameSpec) -> pd.DataFrame:
    """Every cell's drawn records, in quota order, at `frame`'s columns and dtypes."""
    records = [record for records in drawn.values() for record in records]
    rows = pd.DataFrame(records, columns=list(frame.columns))
    return rows.astype({**frame.dtypes})


def label_rows(train: pd.DataFrame, frame: FrameSpec, label: int) -> pd.DataFrame:
    """`train`'s rows whose label is `label`, a cell's label group."""
    return train[train[frame.label] == label]


def native(row: Mapping[Any, Any], frame: FrameSpec) -> Record:
    """`row` with each value a plain `float` for a fractional field and an `int` otherwise.

    An `int` field's value is truncated, so a caller passes one that is integral: the merge's
    has passed `bko.validate`, and a redraw's is a training value.
    """
    return {
        str(name): float(value) if name in frame.fractional else int(value)
        for name, value in row.items()
    }


def _holdout() -> int:
    """The current harness fill's holdout, -1 outside one."""
    fill = CURRENT_FILL.get()
    return -1 if fill is None else fill.holdout


class Perturbing:
    """An arm that draws through a `DirectArm` and perturbs its rows; the inner arm keeps the rest.

    Attributes:
        inner: The direct arm that draws, writes the candidate lines and keeps the counts.
        fields: The fields perturbed.
        counts: The run's holdouts read and empty, and its rows, rows changed and values changed.
        shapes: Each read holdout's perturbed and unperturbed column shapes, in run order.
    """

    def __init__(self, inner: DirectArm, *, fields: Sequence[str], configuration: str) -> None:
        """Hold the arm and log its configuration; nothing is drawn until it runs.

        Args:
            inner: The direct arm that draws.
            fields: The fields perturbed.
            configuration: What the arm's configuration line adds to its fields.
        """
        self.inner = inner
        self.fields = tuple(fields)
        self.counts: Counter[str] = Counter()
        self.shapes: list[tuple[float, float]] = []
        logger.info("perturb arm=%s fields=%s %s", self.name, ",".join(self.fields), configuration)

    @property
    def name(self) -> str:
        """The arm's name."""
        return self.inner.name

    @property
    def totals(self) -> Tally:
        """Every attempt's counts over the run."""
        return self.inner.totals

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The inner arm's drain: its counts, and its lines written."""
        return self.inner.drain(holdout, source)

    def summary(self) -> None:
        """The run's `perturb summary` line, then the inner arm's validity sums."""
        self.summarize("")

    def summarize(self, extra: str) -> None:
        """The run's `perturb summary` line ending `extra`, then the inner arm's validity sums."""
        counts = self.counts
        logger.info(
            "perturb summary arm=%s holdouts=%d read=%d empty=%d rows=%d rows_changed=%d "
            "values_changed=%d mean_shapes=%s mean_unperturbed=%s%s",
            self.name,
            counts["read"] + counts["empty"],
            counts["read"],
            counts["empty"],
            counts["rows"],
            counts["rows_changed"],
            counts["values_changed"],
            _mean([shapes for shapes, _ in self.shapes]),
            _mean([unperturbed for _, unperturbed in self.shapes]),
            extra,
        )
        self.inner.summary()

    def refuse(self, frame: FrameSpec) -> None:
        """Refuse fields that hold `age`, `sex`, `frame`'s label or a field a field row reads.

        Raises:
            EvaluationError: If `fields` holds one.
        """
        mapped = {
            name for group in field_groups(self.inner.rows, frame=frame) for name in group.fields
        }
        held = sorted(set(self.fields) & {"age", "sex", frame.label, *mapped})
        if held:
            msg = f"{self.name} may not perturb {held}: the cell, the label and mapped fields stay"
            raise EvaluationError(msg)

    def read(
        self,
        train: pd.DataFrame,
        drawn: Mapping[Stratum, list[Record]],
        changed: Mapping[Stratum, list[Changed]],
        frame: FrameSpec,
    ) -> None:
        """Log the holdout's reading of `changed` beside `drawn`; it changes no row.

        A fill that drew no row is counted `empty` and logs its `perturb shapes` line with
        `empty=true` alone. `rows_changed` counts the rows whose `result` is in
        `bko.evaluate.holdout_candidates.MOVED`, as their candidates lines' `perturb` names them.
        Each step marks a row moved exactly when one of its fields' values changed, so
        `rows_changed` is above zero exactly when `values_changed` is.
        """
        rows = sum(len(records) for records in drawn.values())
        self.counts["rows"] += rows
        holdout = _holdout()
        if not rows:
            self.counts["empty"] += 1
            logger.info("perturb shapes holdout=%d arm=%s empty=true", holdout, self.name)
            return
        pairs = [
            (row.record, record)
            for cell, originals in drawn.items()
            for row, record in zip(changed[cell], originals, strict=True)
        ]
        moved = sum(row.result in MOVED for held in changed.values() for row in held)
        values = sum(new[name] != old[name] for new, old in pairs for name in self.fields)
        records = {cell: [row.record for row in held] for cell, held in changed.items()}
        reading = read_distribution(
            drawn_frame(records, frame),
            drawn_frame(drawn, frame),
            label_rows(train, frame, POSITIVE),
            fields=self.fields,
            frame=frame,
        )
        self.counts.update(read=1, rows_changed=moved, values_changed=values)
        self.shapes.append((reading.perturbed.shapes, reading.unperturbed.shapes))
        logger.info(
            "perturb shapes holdout=%d arm=%s shapes=%.6f unperturbed=%.6f difference=%.6f "
            "pair_trends=%.6f unperturbed_pair_trends=%.6f incoherent=%s "
            "unperturbed_incoherent=%s rows_changed=%d values_changed=%d",
            holdout,
            self.name,
            reading.perturbed.shapes,
            reading.unperturbed.shapes,
            reading.difference,
            reading.perturbed.pair_trends,
            reading.unperturbed.pair_trends,
            _share(reading.incoherent),
            _share(reading.unperturbed_incoherent),
            moved,
            values,
        )
        for kind, quality_read in (
            ("perturbed", reading.perturbed),
            ("unperturbed", reading.unperturbed),
        ):
            logger.info(
                "perturb columns holdout=%d arm=%s kind=%s %s",
                holdout,
                self.name,
                kind,
                _columns(quality_read),
            )
