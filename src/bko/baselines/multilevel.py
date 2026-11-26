"""The multilevel baseline `hb-marginals`: each column from a hierarchical multinomial logit.

Every feature but `age` and `sex` is drawn column by column from its model (`fit_column` of
`bko.baselines.multilevel_fit`), fitted over every training cell on the column's categories
(`categorise`, over the label group of the cell being filled), its reference category the label
group's modal one; `age` comes from a donor by the pooled rule at `GROUP_WEIGHT`
(`bko.baselines.pooling`), and `sex` and the label from the cell.

A coded column's value is its drawn category, its code. A measured column's, or a coded column's
`OTHER`, is a uniform record of that category from the cell's banded label group, else from its
label group, else from the training split. A fill reads one generator,
`numpy.random.default_rng([request.seed, HB_MARGINALS_STREAM])`, in quota order, record order and
column order: for `age` the pooled rule's coin and record, for another column one uniform for the
category and, when the value is a record's, one integer for the record. It refuses a request by
label, which names no band.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import partial
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.categories import OTHER, categorise
from bko.baselines.fills import fill_names
from bko.baselines.multilevel_fit import Fit, category_order, fit_column
from bko.baselines.pooling import (
    banded_pools,
    drawn_columns,
    pooled_donor,
    refuse_label,
)
from bko.baselines.records import Baseline, cell_rows, joined, unit_weights
from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.cells import LABEL, cell_keys
from bko.gaps.shares import GROUP_WEIGHT


logger = logging.getLogger(__name__)

HB_MARGINALS_STREAM: Final = 20
"""The second word of `hb-marginals`' seed, `[request.seed, HB_MARGINALS_STREAM]`."""

type ColumnDraw = Callable[[np.random.Generator], object]
"""One column's draw of one value from the fill's generator."""


def hb_marginals(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each cell's quota with each column drawn from its multilevel model.

    Each column is fitted once per label of the quota and logs one line per cell it fills.

    Raises:
        BaselineError: If the request is by label, a cell's banded label group is empty, or a fit
            fails (`fit_column`).
    """
    refuse_label(request, "hb-marginals")
    frame = request.frame
    cells = cell_keys(train, edges=frame.edges)
    rng = np.random.default_rng([request.seed, HB_MARGINALS_STREAM])
    models: dict[tuple[int, str], _Column] = {}
    parts: list[pd.DataFrame] = []
    for stratum, rows in request.quota.items():
        drawers = _drawers(train, cells, request, stratum, models)
        records = [[draw(rng) for draw in drawers.values()] for _ in range(rows)]
        part = pd.DataFrame(records, columns=list(drawers), index=range(rows))
        parts.append(cell_rows(part, frame, sex=str(stratum[1]), label=int(stratum[2])))
    added = joined(parts, frame)
    fits = {
        f"{label}/{name}": {
            "tau2": model.fit.tau2,
            "tau3": model.fit.tau3,
            "evidence": model.fit.evidence,
            "iterations": model.fit.iterations,
        }
        for (label, name), model in models.items()
    }
    params = {"stream": HB_MARGINALS_STREAM, "group_weight": GROUP_WEIGHT, "fits": fits}
    return Baseline(added=added, weights=unit_weights(train), params=params)


@dataclass(frozen=True, slots=True, eq=False)
class _Column:
    """One column's fit for one label, with what a draw of a record's value reads.

    Attributes:
        fit: The column's model.
        coded: Whether a category other than `OTHER` is its own value.
        values: The column's training values, by position.
        categories: Each training record's category under the label group's categories.
        label_group: Whether each training record holds the label.
    """

    fit: Fit
    coded: bool
    values: np.ndarray
    categories: np.ndarray
    label_group: np.ndarray

    def __post_init__(self) -> None:
        """Make each array read-only, so no draw changes what later draws read."""
        for array in (self.values, self.categories, self.label_group):
            array.flags.writeable = False

    def draw(self, rng: np.random.Generator, cumulative: np.ndarray, banded: np.ndarray) -> object:
        """One value: a category by `cumulative`, then its code or a record of it."""
        at = min(int(np.searchsorted(cumulative, rng.random(), side="right")), len(cumulative) - 1)
        category = self.fit.categories[at]
        if self.coded and category != OTHER:
            return category
        held = self.categories == category
        pool = banded[held[banded]]
        if not len(pool):
            pool = np.flatnonzero(held & self.label_group)
        if not len(pool):
            pool = np.flatnonzero(held)
        return self.values[pool[int(rng.integers(len(pool)))]]


def _drawers(
    train: pd.DataFrame,
    cells: pd.DataFrame,
    request: Request,
    stratum: Stratum,
    models: dict[tuple[int, str], _Column],
) -> dict[str, ColumnDraw]:
    """Each drawn column's draw in `stratum`, in frame order, with one log line per model.

    `age` draws a donor by the pooled rule; every other column draws from its model for the
    stratum's label, fitted on first use and kept in `models`.

    Raises:
        BaselineError: If the cell's banded label group is empty, or a fit fails.
    """
    key = "/".join(map(str, stratum))
    label = int(stratum[2])
    own, group = banded_pools(cells, stratum, key)
    drawers: dict[str, ColumnDraw] = {}
    for name in drawn_columns(request):
        if name == "age":
            ages = train[name].to_numpy()
            kappa = float(GROUP_WEIGHT)
            drawers[name] = partial(_pooled, values=ages, own=own, group=group, kappa=kappa)
            continue
        if (label, name) not in models:
            models[label, name] = _column(train, cells, request, name, label)
        model = models[label, name]
        fit = model.fit
        logger.info(
            "hb holdout=%d fill=%s key=%s column=%s tau2=%s tau3=%s evidence=%.6f iterations=%d",
            *fill_names(),
            key,
            name,
            fit.tau2,
            fit.tau3,
            fit.evidence,
            fit.iterations,
        )
        cumulative = np.cumsum(fit.predict(stratum))
        drawers[name] = partial(model.draw, cumulative=cumulative, banded=group)
    return drawers


def _pooled(
    rng: np.random.Generator,
    *,
    values: np.ndarray,
    own: np.ndarray,
    group: np.ndarray,
    kappa: float,
) -> object:
    """One value of a donor by the pooled rule (`pooled_donor`)."""
    return values[pooled_donor(rng, own, group, kappa)]


def _column(
    train: pd.DataFrame, cells: pd.DataFrame, request: Request, name: str, label: int
) -> _Column:
    """Fit `name` over every grid cell, categorised over the training records of `label`."""
    frame = request.frame
    inside = (cells[LABEL] == label).to_numpy()
    categories = categorise(train[name], train.loc[inside, name], ordered=name in frame.numeric)
    held = categories[inside]
    modal = held.value_counts()
    reference = min(modal.index[modal == modal.max()], key=category_order)
    counts: dict[Stratum, dict[Hashable, int]] = {}
    for stratum in Strata.CELL.grid(edges=frame.edges):
        band, sex, code = stratum
        rows = (cells["age_band"] == band) & (cells["sex"] == sex) & (cells[LABEL] == code)
        counts[stratum] = {k: int(n) for k, n in categories[rows].value_counts().items()}
    return _Column(
        fit=fit_column(counts, reference=reference),
        coded=name in frame.categorical,
        values=train[name].to_numpy(),
        categories=categories.to_numpy(dtype=object),
        label_group=inside,
    )
