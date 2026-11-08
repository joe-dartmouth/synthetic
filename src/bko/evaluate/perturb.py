"""Perturbing `bko-direct`'s unmapped fields: the fields, delta, the check and the shared arm.

`bko-direct` copies each field no field row reads from its own training record
(`bko.generate.direct`), so its rows are mosaics of training values. `unmapped` names those fields:
every column but the field rows' fields, `age`, `sex` and the label. On the Diabetes frame they
must be `DIABETES_UNMAPPED`, the 19 a rewrite call may send; `unmapped` refuses another set there.
`Perturbing` holds what both perturbation arms share; `bko.evaluate.pram.PramArm` redraws the
fields, and `PerturbedArm` rewrites them through calls merged by `bko.evaluate.rewrite_merge`.

`check_distribution` gates a holdout's perturbed rows: SDMetrics' column shapes of the perturbed
rows against the label-1 training rows, over the unmapped fields, must reach the unperturbed rows'
less delta, or every original row is kept. Column pair trends are logged, not gated. delta is
fixed per removal fraction (`DELTAS`); a fraction listed with None has no value yet, and
`refuse_unfixed` refuses a run that would gate at it, a rehearsal excepted. ADR
`research/identifier-perturbation` records both arms and the check.
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
from bko.datasets.frames import FrameSpec
from bko.evaluate.diabetes import DIABETES_NAME
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import PERTURBED_ARM, PRAM_ARM, Experiment
from bko.evaluate.fidelity import quality_report
from bko.evaluate.scores import Measure, Source
from bko.generate.direct import field_groups, free_fields
from bko.ontology.schema import FieldMapping


if TYPE_CHECKING:
    from bko.evaluate.attempts import Tally
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)

POSITIVE: Final = 1
"""The label whose training rows are the check's reference."""

DELTA_THIN: Final[float | None] = 0.036399
"""delta at E6's thinned fraction, 0.985: the delta reading of run `e10-1`
over E6's stored `bko-direct` rows."""

DELTA_EMPTY: Final[float | None] = 0.003108
"""delta at E6's empty fraction, 1.0: the delta reading of run `e10-2`."""

DELTAS: Final[Mapping[float, float | None]] = MappingProxyType(
    {0.985: DELTA_THIN, 1.0: DELTA_EMPTY}
)
"""Each listed removal fraction's delta, None where no value is recorded."""

PERTURBING_ARMS: Final = frozenset({PERTURBED_ARM, PRAM_ARM})
"""The arms whose rows the distribution check gates."""

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


@dataclass(frozen=True, slots=True)
class Delta:
    """A run's delta and where it comes from.

    Attributes:
        value: The delta; None while no value is recorded for its fraction.
        source: `table` for a fixed `DELTAS` value, `placeholder` for a listed None, `default` for
            a run with no removal or an unlisted fraction, which gates at 0, the strictest.
        fraction: The run's removal fraction; None with no removal.
    """

    value: float | None
    source: str
    fraction: float | None

    @property
    def gate(self) -> float:
        """The delta the check applies: `value`, or 0, the strictest, for a rehearsal at None."""
        return 0.0 if self.value is None else self.value


def delta_for(experiment: Experiment) -> Delta:
    """`experiment`'s delta: its removal fraction's in `DELTAS`, else 0."""
    removal = experiment.removal
    if removal is None:
        return Delta(0.0, "default", None)
    if removal.fraction not in DELTAS:
        return Delta(0.0, "default", removal.fraction)
    value = DELTAS[removal.fraction]
    return Delta(value, "placeholder" if value is None else "table", removal.fraction)


def refuse_unfixed(experiment: Experiment, *, run: str, rehearse: bool) -> None:
    """Refuse a run with a perturbing arm whose delta is not fixed, unless it is a rehearsal.

    Raises:
        EvaluationError: If `experiment` holds a perturbing arm, its delta is None and `rehearse`
            is false.
    """
    arms = sorted(PERTURBING_ARMS & set(experiment.arms))
    delta = delta_for(experiment)
    if arms and delta.value is None and not rehearse:
        msg = (
            f"the run {run!r} holds {', '.join(arms)}, whose delta at removal fraction "
            f"{delta.fraction} is not fixed: record its value in "
            "bko.evaluate.perturb.DELTAS first; only --rehearse runs at an unfixed delta"
        )
        raise EvaluationError(msg)


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
    """

    record: Record
    call: int | None = None
    position: int = 0


type Change = Callable[
    [pd.DataFrame, Request, Mapping[Stratum, list[Record]]], Mapping[Stratum, list[Changed]]
]
"""A perturbation step: from the training rows, the request and each cell's drawn records, in
quota order, each cell's rows in the same order (`bko.evaluate.direct.DirectArm.fill`)."""


def passes(shapes: float, reference: float, delta: float) -> bool:
    """Whether perturbed rows reading `shapes` pass against unperturbed ones reading `reference`.

    They pass at or above `reference - delta`; a shape that is not finite never passes.
    """
    return math.isfinite(shapes) and math.isfinite(reference) and shapes >= reference - delta


@dataclass(frozen=True, slots=True)
class Check:
    """One holdout's distribution check.

    Attributes:
        shapes: The perturbed rows' column shapes against the reference rows.
        reference: The unperturbed rows' column shapes against them.
        pair_trends: The perturbed rows' column pair trends against them, reported only.
        delta: How far below `reference` the perturbed rows may read.
    """

    shapes: float
    reference: float
    pair_trends: float
    delta: float

    @property
    def passed(self) -> bool:
        """Whether the perturbed rows are kept."""
        return passes(self.shapes, self.reference, self.delta)

    @property
    def reason(self) -> str:
        """Why the check refused: `nonfinite` or `below_reference`; `none` when it passed."""
        if self.passed:
            return "none"
        finite = math.isfinite(self.shapes) and math.isfinite(self.reference)
        return "below_reference" if finite else "nonfinite"


def quality(
    rows: pd.DataFrame, reference: pd.DataFrame, *, fields: Sequence[str], frame: FrameSpec
) -> tuple[float, float]:
    """SDMetrics' column shapes and pair trends of `rows` against `reference`, on `fields`."""
    report = quality_report(reference, rows, frame=frame, columns=fields)
    properties = report.get_properties().set_index("Property")["Score"]
    return float(properties["Column Shapes"]), float(properties["Column Pair Trends"])


def check_distribution(
    perturbed: pd.DataFrame,
    original: pd.DataFrame,
    train: pd.DataFrame,
    *,
    fields: Sequence[str],
    frame: FrameSpec,
    delta: float,
) -> Check:
    """Read `perturbed` and `original` against `train`'s label-1 rows over `fields`."""
    reference = train[train[frame.label] == POSITIVE]
    shapes, trends = quality(perturbed, reference, fields=fields, frame=frame)
    unperturbed, _ = quality(original, reference, fields=fields, frame=frame)
    return Check(shapes, unperturbed, trends, delta)


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
        delta: The check's delta and its source.
        counts: The run's holdouts by check outcome, and its rows and rows changed.
    """

    def __init__(
        self, inner: DirectArm, *, fields: Sequence[str], delta: Delta, configuration: str
    ) -> None:
        """Hold the arm and log its configuration; nothing is drawn until it runs.

        Args:
            inner: The direct arm that draws.
            fields: The fields perturbed.
            delta: The check's delta.
            configuration: What the arm's configuration line adds to its fields and delta.
        """
        self.inner = inner
        self.fields = tuple(fields)
        self.delta = delta
        self.counts: Counter[str] = Counter()
        logger.info(
            "perturb arm=%s fields=%s delta=%s delta_source=%s fraction=%s %s",
            self.name,
            ",".join(self.fields),
            "none" if delta.value is None else f"{delta.value:.6f}",
            delta.source,
            "none" if delta.fraction is None else delta.fraction,
            configuration,
        )

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
            "perturb summary arm=%s holdouts=%d accepted=%d refused=%d empty=%d rows=%d "
            "rows_changed=%d%s",
            self.name,
            counts["accepted"] + counts["refused"] + counts["empty"],
            counts["accepted"],
            counts["refused"],
            counts["empty"],
            counts["rows"],
            counts["rows_changed"],
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

    def checked(
        self,
        train: pd.DataFrame,
        drawn: Mapping[Stratum, list[Record]],
        changed: Mapping[Stratum, list[Changed]],
        frame: FrameSpec,
    ) -> dict[Stratum, list[Changed]]:
        """`changed` if it passes the distribution check against `drawn`, else `drawn` unchanged.

        A fill that drew no row has nothing to check, and its rows are `changed`'s.
        """
        rows = sum(len(records) for records in drawn.values())
        self.counts["rows"] += rows
        if not rows:
            self.counts["empty"] += 1
            return dict(changed)
        records = {cell: [row.record for row in found] for cell, found in changed.items()}
        found = check_distribution(
            drawn_frame(records, frame),
            drawn_frame(drawn, frame),
            train,
            fields=self.fields,
            frame=frame,
            delta=self.delta.gate,
        )
        moved = sum(
            row.record != record
            for cell, originals in drawn.items()
            for row, record in zip(changed[cell], originals, strict=True)
        )
        final = moved if found.passed else 0
        self.counts["accepted" if found.passed else "refused"] += 1
        self.counts["rows_changed"] += final
        logger.info(
            "perturb check holdout=%d arm=%s shapes=%.6f reference=%.6f delta=%.6f "
            "pair_trends=%.6f outcome=%s reason=%s rows_changed=%d final=%s",
            _holdout(),
            self.name,
            found.shapes,
            found.reference,
            found.delta,
            found.pair_trends,
            "accepted" if found.passed else "refused",
            found.reason,
            final,
            "perturbed" if found.passed else "original",
        )
        if found.passed:
            return dict(changed)
        return {
            cell: [Changed(dict(record)) for record in records] for cell, records in drawn.items()
        }
