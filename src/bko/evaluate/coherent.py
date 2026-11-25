"""`bko-coherent`: `bko-direct`'s draw with its free fields redrawn by sequential CART.

`CoherentArm` draws exactly as `bko-direct` does, then redraws the free fields
(`bko.evaluate.perturb.unmapped`) by `bko.baselines.conditional.conditional_cart`: one tree per
free field, fit once per fill on the holdout's whole training split, given `age`, `sex`, the label
and the mapped fields in frame order and the free fields visited before it, in frame order. Each
value is copied from a training row drawn uniformly from the new row's leaf, its leaf donor. The
leaf draws come from `numpy.random.default_rng([seed, COHERENT_STREAM])` at the request's seed, a
stream apart from `bko-direct`'s draw and from PRAM's. Each row's candidates line's `perturb` says
whether its record differs from its draw (`changed`) or not (`unchanged`). It makes no call.
`coherent_records` is the one redraw both the arm and `coherent_draws`, which `trace` and donor
reuse read, run. ADR `research/free-field-synthesis` records the arm.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from bko.baselines.conditional import conditional_cart
from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline
from bko.baselines.request import Request, Stratum
from bko.evaluate.holdout_candidates import CHANGED, UNCHANGED
from bko.evaluate.perturb import Changed, Perturbing, Record, native
from bko.generate.direct import Draw


if TYPE_CHECKING:
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)

COHERENT_STREAM: Final = 2
"""The second word of the leaf draws' seed, `[request.seed, COHERENT_STREAM]`, which sets their
stream apart from `bko-direct`'s draw at `request.seed` and from PRAM's
(`bko.evaluate.pram.PRAM_STREAM`)."""


def coherent_records(
    train: pd.DataFrame, request: Request, records: Sequence[Record], *, fields: Sequence[str]
) -> tuple[list[Record], list[dict[str, int]]]:
    """`records`, in order, with `fields` redrawn by conditional CART, and each one's leaf donors.

    Every other column of the frame is a given predictor, in frame order; `fields` are visited
    in frame order. The trees fit on `train`, seeded by the request's seed, and the leaf draws read
    `default_rng([request.seed, COHERENT_STREAM])`.

    Returns:
        Each record redrawn, and the training record id each of its `fields` was copied from.

    Raises:
        ValueError: If an entry of `fields` is not a column of the request's frame.
    """
    frame = request.frame
    for name in fields:
        if name not in frame.columns:
            msg = f"field {name!r} is not a column of frame {frame.name!r}"
            raise ValueError(msg)
    visit = [name for name in frame.columns if name in fields]
    given = [name for name in frame.columns if name not in fields]
    rows, donors = conditional_cart(
        train,
        pd.DataFrame(list(records), columns=list(frame.columns)).astype({**frame.dtypes}),
        given=given,
        visit=visit,
        frame=frame,
        rng=np.random.default_rng([request.seed, COHERENT_STREAM]),
        seed=request.seed,
    )
    redrawn = [native(row, frame) for row in rows.to_dict("records")]
    ids = [{str(name): int(at) for name, at in row.items()} for row in donors.to_dict("records")]
    return redrawn, ids


def coherent_draws(
    train: pd.DataFrame, request: Request, draws: Sequence[Draw], *, fields: Sequence[str]
) -> list[Draw]:
    """Each of `draws`, in generation order, as the arm changes it: `coherent_records` of them.

    Each draw's record is its redrawn one, each of `fields` names its leaf donor as its source,
    and none of `fields` is a fallback field.
    """
    records, donors = coherent_records(
        train, request, [dict(draw.record) for draw in draws], fields=fields
    )
    return [
        dataclasses.replace(
            draw,
            record=record,
            sources={**draw.sources, **ids},
            fallback=draw.fallback - set(fields),
        )
        for draw, record, ids in zip(draws, records, donors, strict=True)
    ]


class CoherentArm(Perturbing):
    """`bko-coherent`: `bko-direct`'s draw with its free fields redrawn by sequential CART."""

    def __init__(self, inner: DirectArm, *, fields: Sequence[str]) -> None:
        """Hold the arm and log its configuration; nothing is drawn until it runs."""
        super().__init__(inner, fields=fields, configuration=f"stream={COHERENT_STREAM}")

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct` and redraw the free fields, seeded by the request's seed.

        Raises:
            EvaluationError: As `bko.evaluate.perturb.Perturbing.refuse`.
            As `bko.evaluate.direct.DirectArm.fill`.
        """
        self.refuse(request.frame)
        params = {"perturb": "coherent"}
        return self.inner.fill(train, request, change=self._change, params=params)

    def _change(
        self, train: pd.DataFrame, request: Request, drawn: Mapping[Stratum, list[Record]]
    ) -> dict[Stratum, list[Changed]]:
        frame = request.frame
        flat = [record for records in drawn.values() for record in records]
        rows, _ = coherent_records(train, request, flat, fields=self.fields)
        changed: dict[Stratum, list[Changed]] = {}
        start = 0
        for cell, records in drawn.items():
            news = rows[start : start + len(records)]
            changed[cell] = [
                Changed(new, result=CHANGED if old != new else UNCHANGED)
                for old, new in zip(records, news, strict=True)
            ]
            start += len(records)
        fill = CURRENT_FILL.get()
        logger.info(
            "perturb coherent holdout=%d arm=%s rows=%d",
            -1 if fill is None else fill.holdout,
            self.name,
            len(rows),
        )
        self.read(train, drawn, changed, frame)
        return changed
