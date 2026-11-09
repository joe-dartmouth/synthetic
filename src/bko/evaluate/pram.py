"""`bko-direct-pram`: `bko-direct`'s draw with its unmapped fields perturbed by PRAM.

`PramArm` draws exactly as `bko-direct` does, then perturbs the unmapped fields
(`bko.evaluate.perturb.unmapped`) by PRAM: each value is kept with probability `THETA`, otherwise
redrawn from the training rows of its row's label, so each field redraws toward the label-group
training marginal. A field's value and the row's other fields are redrawn independently, so a
redrawn row need not keep its fields' joint combination. Its draws come from
`numpy.random.default_rng([seed, PRAM_STREAM])` at the request's seed, a stream apart from
`bko-direct`'s own at that seed, field by field in column order. Each row's candidates line's
`perturb` says whether its record differs from its draw (`changed`) or not (`unchanged`). It makes
no call. Mapped fields, `age`, `sex` and the label keep their drawn values. ADR
`research/identifier-perturbation` records the arm.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline
from bko.baselines.request import Request, Stratum
from bko.datasets.frames import FrameSpec
from bko.evaluate.holdout_candidates import CHANGED, UNCHANGED
from bko.evaluate.perturb import Changed, Perturbing, Record, drawn_frame, native


if TYPE_CHECKING:
    from bko.evaluate.direct import DirectArm


logger = logging.getLogger(__name__)

THETA: Final = 0.5
"""The probability PRAM keeps a value."""

PRAM_STREAM: Final = 1
"""The second word of PRAM's seed, `[request.seed, PRAM_STREAM]`, which sets its stream apart
from `bko-direct`'s draw at `request.seed`."""


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
    each field redraws toward that label group's training marginal; a redraw may equal the value
    it replaces. For each field in turn, `rng` draws whether each row keeps its value, then which
    training value it would take. A row whose label no training row holds keeps every value.
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


class PramArm(Perturbing):
    """`bko-direct-pram`: `bko-direct`'s draw with its unmapped fields perturbed by PRAM.

    Attributes:
        theta: The probability a value is kept.
    """

    def __init__(self, inner: DirectArm, theta: float = THETA, *, fields: Sequence[str]) -> None:
        """Hold the arm and log its configuration; nothing is drawn until it runs.

        Raises:
            ValueError: If `theta` is outside [0, 1].
        """
        if not 0 <= theta <= 1:
            msg = f"theta {theta} must lie in [0, 1]"
            raise ValueError(msg)
        self.theta = theta
        super().__init__(inner, fields=fields, configuration=f"theta={theta}")

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Draw as `bko-direct` and perturb, seeded by the request's seed.

        Raises:
            EvaluationError: As `bko.evaluate.perturb.Perturbing.refuse`.
            As `bko.evaluate.direct.DirectArm.fill`.
        """
        self.refuse(request.frame)
        params = {"perturb": "pram", "theta": self.theta}
        return self.inner.fill(train, request, change=self._change, params=params)

    def _change(
        self, train: pd.DataFrame, request: Request, drawn: Mapping[Stratum, list[Record]]
    ) -> dict[Stratum, list[Changed]]:
        frame = request.frame
        rng = np.random.default_rng([request.seed, PRAM_STREAM])
        original = drawn_frame(drawn, frame)
        perturbed = pram(
            original, train, fields=self.fields, theta=self.theta, rng=rng, frame=frame
        )
        rows = perturbed.to_dict("records")
        changed: dict[Stratum, list[Changed]] = {}
        start = 0
        for cell, records in drawn.items():
            news = [native(row, frame) for row in rows[start : start + len(records)]]
            changed[cell] = [
                Changed(new, result=CHANGED if old != new else UNCHANGED)
                for old, new in zip(records, news, strict=True)
            ]
            start += len(records)
        fill = CURRENT_FILL.get()
        logger.info(
            "perturb pram holdout=%d arm=%s rows=%d",
            -1 if fill is None else fill.holdout,
            self.name,
            len(rows),
        )
        self.read(train, drawn, changed, frame)
        return changed
