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
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline
from bko.baselines.request import Request, Stratum
from bko.datasets.cells import SEXES
from bko.datasets.frames import FrameSpec
from bko.evaluate.experiment import Experiment
from bko.evaluate.fidelity import quality_report
from bko.evaluate.scores import Measure, Source
from bko.generate.calls import chunks
from bko.generate.decode import Undecoded
from bko.generate.direct import field_groups
from bko.generate.flight import Flight, Planned
from bko.generate.guidance import Guidance
from bko.generate.rewrite import ROW, Rewrite
from bko.ontology.schema import FieldMapping
from bko.validate.checks import failures


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


def native(row: Mapping[Hashable, Any] | Mapping[str, Any], frame: FrameSpec) -> Record:
    """`row` with each value a plain `float` for a fractional field and an `int` otherwise."""
    return {
        str(name): float(value) if name in frame.fractional else int(value)
        for name, value in row.items()
    }


class Perturbing:
    """An arm that draws through a `DirectArm` and perturbs its rows; the inner arm keeps the rest.

    Attributes:
        inner: The direct arm that draws, writes the candidate lines and keeps the counts.
        fields: The fields perturbed.
        delta: The check's δ.
    """

    def __init__(self, inner: DirectArm, *, fields: Sequence[str], delta: float) -> None:
        """Hold the arm; nothing is drawn until it runs."""
        self.inner = inner
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

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The inner arm's drain: its counts, and its lines written."""
        return self.inner.drain(holdout, source)

    def summary(self) -> None:
        """The inner arm's validity sums."""
        self.inner.summary()

    def checked(
        self,
        train: pd.DataFrame,
        drawn: Mapping[Stratum, list[Record]],
        changed: Mapping[Stratum, list[Changed]],
        frame: FrameSpec,
    ) -> dict[Stratum, list[Changed]]:
        """`changed` if it passes the distribution check against `drawn`, else `drawn` unchanged."""
        original = drawn_frame(drawn, frame)
        rows = {cell: [row.record for row in found] for cell, found in changed.items()}
        found = check_distribution(
            drawn_frame(rows, frame),
            original,
            train,
            fields=self.fields,
            frame=frame,
            delta=self.delta,
        )
        log_check(found, arm=self.name)
        if found.passed:
            return dict(changed)
        return {
            cell: [Changed(dict(record)) for record in records] for cell, records in drawn.items()
        }


class PramArm(Perturbing):
    """`bko-direct-pram`: `bko-direct`'s draw with its unmapped fields perturbed by PRAM.

    Attributes:
        theta: The probability a value is kept.
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
        super().__init__(inner, fields=fields, delta=delta)
        self.theta = theta

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct` and perturb, seeded by the request's seed.

        Raises:
            As `bko.evaluate.direct.DirectArm.fill`.
        """
        params = {"perturb": "pram", "theta": self.theta}
        return self.inner.fill(train, request, change=self._change, params=params)

    def _change(
        self, train: pd.DataFrame, request: Request, drawn: Mapping[Stratum, list[Record]]
    ) -> dict[Stratum, list[Changed]]:
        frame = request.frame
        rng = np.random.default_rng(request.seed)
        original = drawn_frame(drawn, frame)
        perturbed = pram(
            original, train, fields=self.fields, theta=self.theta, rng=rng, frame=frame
        ).to_dict("records")
        changed: dict[Stratum, list[Changed]] = {}
        start = 0
        for cell, records in drawn.items():
            rows = perturbed[start : start + len(records)]
            changed[cell] = [Changed(native(row, frame)) for row in rows]
            start += len(records)
        return self.checked(train, drawn, changed, frame)


@dataclass(frozen=True, slots=True)
class Merged:
    """One rewrite call's reply merged into its rows by `row`.

    Attributes:
        rows: Each row asked, in order: its original with the reply's fields, or its original.
        outcomes: Each row's: `perturbed` or `unchanged` when it took the reply's fields,
            `invalid` when its reply failed a check and it kept its original, or `rows` when the
            reply's row indices were not those asked, so every row kept its original.
        returned: The rows the reply decoded to.
    """

    rows: tuple[Record, ...]
    outcomes: tuple[str, ...]
    returned: int

    @property
    def reason(self) -> str:
        """Why a row kept its original: `rows`, then `invalid`; `none` when none did."""
        return next((r for r in ("rows", "invalid") if r in self.outcomes), "none")


def merge(
    items: Sequence[object],
    originals: Sequence[Record],
    *,
    fields: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    frame: FrameSpec,
) -> Merged:
    """Merge a rewrite call's decoded `items` into `originals` by `row`.

    The reply's `row` indices must be the indices asked, in order, each an `int`; otherwise every
    row keeps its original. A row's reply must hold `row` and each of `fields` and no other key,
    and the merged row must pass `bko.validate` and lie within `bounds`; otherwise that row keeps
    its original. So mapped fields, `age`, `sex` and the label never change.
    """
    decoded = [item for item in items if not isinstance(item, Undecoded)]
    indices = [item.get(ROW) if isinstance(item, Mapping) else None for item in decoded]
    if any(type(index) is not int for index in indices) or indices != list(range(len(originals))):
        return Merged(tuple(originals), ("rows",) * len(originals), len(decoded))
    rows: list[Record] = []
    outcomes: list[str] = []
    for item, original in zip(decoded, originals, strict=True):
        merged = _merged(item, original, fields=fields, bounds=bounds, frame=frame)
        rows.append(dict(original) if merged is None else merged)
        if merged is None:
            outcomes.append("invalid")
        else:
            outcomes.append("unchanged" if merged == original else "perturbed")
    return Merged(tuple(rows), tuple(outcomes), len(decoded))


def _merged(
    item: object,
    original: Record,
    *,
    fields: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    frame: FrameSpec,
) -> Record | None:
    """`original` with `item`'s fields, or None if `item` fails a check."""
    if not isinstance(item, Mapping) or set(item) != {ROW, *fields}:
        return None
    record = {**original, **{name: item[name] for name in fields}}
    if failures(record, frame=frame):
        return None
    if any(not low <= record[name] <= high for name, (low, high) in bounds.items()):
        return None
    return native(record, frame)


class PerturbedArm(Perturbing):
    """`bko-direct-perturbed`: `bko-direct`'s draw with its unmapped fields rewritten by calls.

    Attributes:
        flight: The run's `Flight`, over its send and call log, shared with every paid arm.
    """

    def __init__(
        self, inner: DirectArm, *, flight: Flight, fields: Sequence[str], delta: float
    ) -> None:
        """Hold the arm; nothing is drawn or sent until it runs."""
        super().__init__(inner, fields=fields, delta=delta)
        self.flight = flight

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct`, then rewrite each row's fields through the run's calls.

        Raises:
            As `bko.evaluate.direct.DirectArm.fill` and `bko.generate.flight.Flight.wave`.
        """
        params = {"perturb": "rewrite"}
        return self.inner.fill(train, request, change=self._change, params=params)

    def _change(
        self, train: pd.DataFrame, request: Request, drawn: Mapping[Stratum, list[Record]]
    ) -> dict[Stratum, list[Changed]]:
        """Send each cell's rows in chunks of `per_call`, one wave, and merge each reply."""
        frame = request.frame
        fill = CURRENT_FILL.get()
        holdout = -1 if fill is None else fill.holdout
        bounds = {
            name: (float(train[name].min()), float(train[name].max()))
            for name in self.fields
            if name not in frame.categorical
        }
        per_call = self.flight.log.settings.per_call
        planned: list[Planned] = []
        spans: list[tuple[Stratum, int, list[Record]]] = []
        starts = dict.fromkeys(drawn, 0)
        for cell, count in chunks({cell: len(rows) for cell, rows in drawn.items()}, per_call):
            start, starts[cell] = starts[cell], starts[cell] + count
            rows = drawn[cell][start : start + count]
            sent = tuple({name: row[name] for name in self.fields} for row in rows)
            rewrite = Rewrite(self.fields, sent, bounds)
            planned.append(Planned(_guidance(cell), count, rewrite=rewrite))
            spans.append((cell, start, rows))
        batches = self.flight.wave(planned, frame=frame) if planned else ()
        changed = {cell: [Changed(dict(row)) for row in rows] for cell, rows in drawn.items()}
        counts: Counter[str] = Counter()
        for batch, (cell, start, rows) in zip(batches, spans, strict=True):
            merged = merge(batch.candidates, rows, fields=self.fields, bounds=bounds, frame=frame)
            counts.update(merged.outcomes)
            for offset, (row, outcome) in enumerate(zip(merged.rows, merged.outcomes, strict=True)):
                if outcome in {"perturbed", "unchanged"}:
                    changed[cell][start + offset] = Changed(row, batch.index, offset)
            refused = sum(o in {"rows", "invalid"} for o in merged.outcomes)
            logger.info(
                "perturb call=%d holdout=%d rows=%d returned=%d kept=%d refused=%d reason=%s",
                batch.index,
                holdout,
                len(rows),
                merged.returned,
                len(rows) - refused,
                refused,
                merged.reason,
            )
        logger.info(
            "perturb holdout=%d rows=%d perturbed=%d unchanged=%d refused_rows=%d "
            "refused_invalid=%d",
            holdout,
            sum(len(rows) for rows in drawn.values()),
            counts["perturbed"],
            counts["unchanged"],
            counts["rows"],
            counts["invalid"],
        )
        return self.checked(train, drawn, changed, frame)


def _guidance(cell: Stratum) -> Guidance:
    """A rewrite call's guidance: only its cell, `sex` and label, which `request` reads."""
    sexes = {meaning: code for code, meaning in SEXES.items()}
    return Guidance(cell=cell, sex=sexes[str(cell[1])], label=int(cell[2]), records=0, interval={})
