"""Perturbing `bko-direct`'s unmapped fields: invariant PRAM and the per-holdout distribution check.

`bko-direct` copies each field no field row reads from its own training record
(`bko.generate.direct`), so its rows are mosaics of training values. `unmapped` names those fields:
every column but the field rows' fields, `age`, `sex` and the label, the 19 of Diabetes 130.
`PramArm` draws exactly as `bko-direct` does, then perturbs those fields by invariant PRAM: each
value is kept with probability `theta`, otherwise redrawn from the training rows of its row's label,
so each field's marginal over the label group is invariant in expectation. Its draws come from one
`numpy.random.default_rng` at the request's seed, field by field in column order. It makes no
call. Mapped fields, `age`, `sex` and the label keep their drawn values.

`check_distribution` gates a holdout's perturbed rows: SDMetrics' column shapes of the perturbed
rows against the label-1 training rows, over the unmapped fields, must reach the unperturbed rows'
less δ, or every original row is kept. Column pair trends are logged, not gated. δ is fixed per
removal fraction (`DELTAS`) by `perturb-threshold` (`bko.evaluate.perturb_threshold`). ADR
`research/identifier-perturbation` records both arms and the check.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline
from bko.baselines.request import Request, Stratum
from bko.datasets.frames import FrameSpec
from bko.evaluate.experiment import Experiment
from bko.evaluate.fidelity import quality_report
from bko.evaluate.scores import Measure, Source
from bko.generate.direct import field_groups
from bko.ontology.schema import FieldMapping


if TYPE_CHECKING:
    from bko.evaluate.attempts import Tally
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)

THETA: Final = 0.5
"""The probability PRAM keeps a value."""

POSITIVE: Final = 1
"""The label whose training rows are the check's reference."""

DELTA_THIN: Final = 0.0
"""δ at E6's thinned fraction, 0.985: not yet fixed; `perturb-threshold` reads it from E6's
stored `bko-direct` rows."""

DELTA_EMPTY: Final = 0.0
"""δ at E6's empty fraction, 1.0: not yet fixed, as `DELTA_THIN`."""

DELTAS: Final[Mapping[float, float]] = MappingProxyType({0.985: DELTA_THIN, 1.0: DELTA_EMPTY})
"""Each removal fraction's δ."""

type Record = dict[str, int | float]
"""One drawn row's value per column."""


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


def delta_for(experiment: Experiment) -> float:
    """`experiment`'s δ: its removal fraction's in `DELTAS`, else 0, the strictest."""
    removal = experiment.removal
    return 0.0 if removal is None else DELTAS.get(removal.fraction, 0.0)


def unmapped(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> tuple[str, ...]:
    """`frame`'s columns that no field row reads, but `age`, `sex` and the label, in frame order."""
    grouped = {name for group in field_groups(rows, frame=frame) for name in group.fields}
    cell = {"age", "sex", frame.label}
    return tuple(name for name in frame.columns if name not in grouped and name not in cell)


def pram(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    *,
    fields: Sequence[str],
    theta: float,
    rng: np.random.Generator,
    frame: FrameSpec,
) -> pd.DataFrame:
    """`rows` with each of `fields` kept with probability `theta`, else redrawn from `train`.

    A redrawn value is a uniform draw from the values of `train`'s rows with the row's label, so
    each field's marginal over that label group is invariant in expectation. For each field in
    turn, `rng` draws whether each row keeps its value, then which training value it would take.
    A row whose label no training row holds keeps every value.
    """
    out = rows.copy()
    labels = rows[frame.label].to_numpy()
    groups = {label: train[train[frame.label] == label] for label in np.unique(labels)}
    for name in fields:
        keep = rng.random(len(rows)) < theta
        pick = rng.random(len(rows))
        values = out[name].to_numpy(copy=True)
        for label, group in groups.items():
            pool = group[name].to_numpy()
            redrawn = (labels == label) & ~keep
            if len(pool):
                values[redrawn] = pool[(pick[redrawn] * len(pool)).astype(int)]
        out[name] = values
    return out


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


def log_check(found: Check, *, arm: str) -> None:
    """Log `found` for the current fill's holdout, -1 outside a harness fill."""
    fill = CURRENT_FILL.get()
    logger.info(
        "perturb check holdout=%d arm=%s shapes=%.6f reference=%.6f delta=%.6f "
        "pair_trends=%.6f outcome=%s",
        -1 if fill is None else fill.holdout,
        arm,
        found.shapes,
        found.reference,
        found.delta,
        found.pair_trends,
        "accepted" if found.passed else "refused",
    )


def drawn_frame(drawn: Mapping[Stratum, list[Record]], frame: FrameSpec) -> pd.DataFrame:
    """Every cell's drawn records, in quota order, at `frame`'s columns and dtypes."""
    records = [record for records in drawn.values() for record in records]
    rows = pd.DataFrame(records, columns=list(frame.columns))
    return rows.astype({**frame.dtypes})


def native(row: Mapping[Hashable, Any], frame: FrameSpec) -> Record:
    """`row` with each value a plain `float` for a fractional field and an `int` otherwise."""
    return {
        str(name): float(value) if name in frame.fractional else int(value)
        for name, value in row.items()
    }


class PramArm:
    """`bko-direct-pram`: `bko-direct`'s draw with its unmapped fields perturbed by PRAM.

    Attributes:
        inner: The direct arm that draws, writes the candidate lines and keeps the counts.
        theta: The probability a value is kept.
        fields: The fields perturbed.
        delta: The check's δ.
    """

    def __init__(
        self, inner: DirectArm, theta: float = THETA, *, fields: Sequence[str], delta: float
    ) -> None:
        """Hold the arm; nothing is drawn until it runs.

        Raises:
            ValueError: If `theta` is outside [0, 1].
        """
        if not 0 <= theta <= 1:
            msg = f"theta {theta} must lie in [0, 1]"
            raise ValueError(msg)
        self.inner = inner
        self.theta = theta
        self.fields = tuple(fields)
        self.delta = delta

    @property
    def name(self) -> str:
        """The arm's name."""
        return self.inner.name

    @property
    def totals(self) -> Tally:
        """Every attempt's counts over the run."""
        return self.inner.totals

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct` and perturb, seeded by the request's seed.

        Raises:
            As `bko.evaluate.direct.DirectArm.fill`.
        """
        params = {"perturb": "pram", "theta": self.theta}
        return self.inner.fill(train, request, change=self._change, params=params)

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The inner arm's drain: its counts, and its lines written."""
        return self.inner.drain(holdout, source)

    def summary(self) -> None:
        """The inner arm's validity sums."""
        self.inner.summary()

    def _change(
        self, train: pd.DataFrame, request: Request, drawn: Mapping[Stratum, list[Record]]
    ) -> dict[Stratum, list[Changed]]:
        frame = request.frame
        original = drawn_frame(drawn, frame)
        rng = np.random.default_rng(request.seed)
        perturbed = pram(
            original, train, fields=self.fields, theta=self.theta, rng=rng, frame=frame
        )
        found = check_distribution(
            perturbed, original, train, fields=self.fields, frame=frame, delta=self.delta
        )
        log_check(found, arm=self.name)
        kept = (perturbed if found.passed else original).to_dict("records")
        out: dict[Stratum, list[Changed]] = {}
        start = 0
        for cell, records in drawn.items():
            out[cell] = [Changed(native(row, frame)) for row in kept[start : start + len(records)]]
            start += len(records)
        return out
