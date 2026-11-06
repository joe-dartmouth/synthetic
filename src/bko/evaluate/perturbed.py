"""`bko-direct-perturbed`: `bko-direct`'s rows with their unmapped fields rewritten by calls.

`PerturbedArm` draws exactly as `bko-direct` does, then sends each cell's rows, only the unmapped
fields (`bko.evaluate.perturb.unmapped`) with a `row` index each, in chunks of `per_call` rows, one
wave of rewrite calls (`bko.generate.rewrite`) per holdout, and merges each reply by `row`.

`merge` takes a call's decoded items (`bko.generate.decode.decode`) and the rows the call sent,
whole, and returns each row as it ends up and why. A call's reply is refused whole, every row
keeping its original, when it decoded to nothing (`empty`), to another count of items than rows
sent (`short`), to an item that is not an object with an integer `row` (`undecodable`), or to
`row` indices other than those asked, in order (`order`). Otherwise each row is merged alone and
refused, keeping its original, when its item holds other keys than `row` and the fields
(`keys`, the harness's key-set check), the merged row fails `bko.validate` (`validate`, with the
checks it failed), or a measured field leaves the training range (`bounds`). A merged row takes
its fields' values as `bko.evaluate.perturb.native` writes them: an integral `float` of a
whole-number field becomes its `int`, since `bko.validate` has refused any other. ADR
`research/identifier-perturbation` records the arm and the merge.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import pandas as pd

from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline
from bko.baselines.request import Request, Stratum
from bko.datasets.cells import SEXES
from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError
from bko.evaluate.perturb import Changed, Delta, Perturbing, Record, native
from bko.generate.calls import chunks
from bko.generate.decode import Undecoded
from bko.generate.flight import Flight, Planned
from bko.generate.guidance import Guidance
from bko.generate.rewrite import ROW, Rewrite
from bko.validate.checks import failures


if TYPE_CHECKING:
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)


TAKEN: Final = ("perturbed", "unchanged")
"""The outcomes of a row that took its reply's fields."""

CALL_REFUSALS: Final = ("empty", "short", "undecodable", "order")
"""Why a whole call's rows kept their originals, in the order `merge` tests them."""

ROW_REFUSALS: Final = ("keys", "validate", "bounds")
"""Why one row kept its original, in the order `merge` tests them."""

OUTCOMES: Final = (*TAKEN, *CALL_REFUSALS, *ROW_REFUSALS)
"""Every row outcome `merge` names."""


@dataclass(frozen=True, slots=True)
class Merged:
    """One rewrite call's reply merged into its rows by `row`.

    Attributes:
        rows: Each row asked, in order: its original with the reply's fields, or its original.
        outcomes: Each row's, one of `OUTCOMES`.
        returned: The items the reply decoded to.
        checks: The `bko.validate` checks the call's `validate` rows failed, sorted.
    """

    rows: tuple[Record, ...]
    outcomes: tuple[str, ...]
    returned: int
    checks: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        """Why a row kept its original, the first refusal in `OUTCOMES`; `none` when none did."""
        refusals = (*CALL_REFUSALS, *ROW_REFUSALS)
        return next((r for r in refusals if r in self.outcomes), "none")

    @property
    def refused(self) -> int:
        """The rows that kept their originals."""
        return sum(outcome not in TAKEN for outcome in self.outcomes)


def merge(
    items: Sequence[object],
    originals: Sequence[Record],
    *,
    fields: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    frame: FrameSpec,
) -> Merged:
    """Merge a rewrite call's decoded `items` into `originals` by `row`, as the module says.

    Mapped fields, `age`, `sex` and the label are never among `fields`, so they never change.
    """
    decoded = [item for item in items if not isinstance(item, Undecoded)]
    refusal = _call_refusal(decoded, len(originals))
    if refusal is not None:
        return Merged(tuple(originals), (refusal,) * len(originals), len(decoded))
    rows: list[Record] = []
    outcomes: list[str] = []
    checks: set[str] = set()
    for item, original in zip(cast("list[Mapping[str, object]]", decoded), originals, strict=True):
        record = {**original, **{name: item[name] for name in fields if name in item}}
        failed = () if set(item) != {ROW, *fields} else failures(record, frame=frame)
        merged = native(record, frame) if set(item) == {ROW, *fields} and not failed else original
        checks.update(failed)
        if set(item) != {ROW, *fields}:
            outcome = "keys"
        elif failed:
            outcome = "validate"
        elif any(not low <= merged[name] <= high for name, (low, high) in bounds.items()):
            outcome, merged = "bounds", original
        else:
            outcome = "unchanged" if merged == original else "perturbed"
        rows.append(dict(merged))
        outcomes.append(outcome)
    return Merged(tuple(rows), tuple(outcomes), len(decoded), tuple(sorted(checks)))


def _call_refusal(decoded: Sequence[object], asked: int) -> str | None:
    """Why the whole call is refused, or None when its items align with the rows asked."""
    if not decoded:
        return "empty"
    if len(decoded) != asked:
        return "short"
    indices = [item.get(ROW) if isinstance(item, Mapping) else None for item in decoded]
    if any(type(index) is not int for index in indices):
        return "undecodable"
    return None if indices == list(range(asked)) else "order"


class PerturbedArm(Perturbing):
    """`bko-direct-perturbed`: `bko-direct`'s draw with its unmapped fields rewritten by calls.

    Attributes:
        flight: The run's `Flight`, over its send and call log, shared with every paid arm.
        merged: The run's rows by merge outcome.
    """

    def __init__(
        self, inner: DirectArm, *, flight: Flight, fields: Sequence[str], delta: Delta
    ) -> None:
        """Hold the arm and log its configuration; nothing is drawn or sent until it runs."""
        self.flight = flight
        self.merged: Counter[str] = Counter()
        settings = flight.log.settings
        configuration = f"per_call={settings.per_call} model={settings.model}"
        super().__init__(inner, fields=fields, delta=delta, configuration=configuration)

    def summary(self) -> None:
        """The run's `perturb summary` line with its rows by merge outcome (`counted`)."""
        self.summarize(counted(self.merged))

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct`, then rewrite each row's fields through the run's calls.

        Raises:
            EvaluationError: As `bko.evaluate.perturb.Perturbing.refuse`, or for a cell whose sex
                no code names.
            As `bko.evaluate.direct.DirectArm.fill` and `bko.generate.flight.Flight.wave`.
        """
        self.refuse(request.frame)
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
                if outcome in TAKEN:
                    changed[cell][start + offset] = Changed(row, batch.index, offset)
            logger.info(
                "perturb call=%d holdout=%d rows=%d returned=%d accepted=%d refused=%d reason=%s "
                "stop_reason=%s checks=%s",
                batch.index,
                holdout,
                len(rows),
                merged.returned,
                len(rows) - merged.refused,
                merged.refused,
                merged.reason,
                batch.reply.stop_reason,
                ",".join(merged.checks) or "none",
            )
        self.merged.update(counts)
        logger.info(
            "perturb holdout=%d rows=%d%s",
            holdout,
            sum(len(rows) for rows in drawn.values()),
            counted(counts),
        )
        return self.checked(train, drawn, changed, frame)


def counted(counts: Mapping[str, int]) -> str:
    """Each of `OUTCOMES`' row count in `counts`, a refusal's key prefixed `refused_`."""
    return "".join(
        f" {'' if outcome in TAKEN else 'refused_'}{outcome}={counts.get(outcome, 0)}"
        for outcome in OUTCOMES
    )


def _guidance(cell: Stratum) -> Guidance:
    """A rewrite call's guidance: only its cell, `sex` and label, which `request` reads.

    Raises:
        EvaluationError: If no code names the cell's sex.
    """
    sexes = {meaning: code for code, meaning in SEXES.items()}
    sex = sexes.get(str(cell[1]))
    if sex is None:
        msg = f"no sex code names cell {cell}'s sex {cell[1]!r}; the codes name {sorted(sexes)}"
        raise EvaluationError(msg)
    return Guidance(cell=cell, sex=sex, label=int(cell[2]), records=0, interval={})
