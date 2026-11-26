"""The pooled retained-data methods: `row-resample`, `marginals` and `eb-marginals`.

Each fills a cell from the training records of its label in its age band, its banded label group,
by the pooled rule: a donor is a uniform record of the cell itself with probability `n / (n +
kappa)`, for the cell's `n` training records, else a uniform record of the banded label group,
the cell's own records among them. A donor from the band keeps the row in its cell, and the row
takes the cell's sex. `row-resample` copies a whole row from one donor at `kappa` `GROUP_WEIGHT`;
`marginals` draws each column from its own donor, which breaks every within-record association;
`eb-marginals` does so at one empirical-Bayes `kappa` for every column, `fit_kappa`'s maximum.

A fill reads one generator, `numpy.random.default_rng([request.seed, STREAM])` with the method's
stream constant, in quota order, record order and column order: per donor one uniform for the coin,
then one integer for the record. Each refuses a request by label, which names no band.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import gammaln

from bko.baselines.categories import categorise
from bko.baselines.errors import BaselineError
from bko.baselines.fills import fill_names
from bko.baselines.records import Baseline, cell_rows, joined, unit_weights
from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.cells import LABEL, SEXES, age_band_labels, cell_keys
from bko.gaps.shares import GROUP_WEIGHT, cell_weight


logger = logging.getLogger(__name__)

ROW_RESAMPLE_STREAM: Final = 16
"""The second word of `row-resample`'s seed, `[request.seed, ROW_RESAMPLE_STREAM]`."""

MARGINALS_STREAM: Final = 17
"""The second word of `marginals`' seed, `[request.seed, MARGINALS_STREAM]`."""

EB_MARGINALS_STREAM: Final = 18
"""The second word of `eb-marginals`' seed, `[request.seed, EB_MARGINALS_STREAM]`."""

KAPPA_GRID: Final[tuple[float, ...]] = tuple(float(kappa) for kappa in np.logspace(-1, 4, 101))
"""The weights `fit_kappa` chooses among: 101 log-spaced values from 0.1 to 10000."""

type Term[K: Hashable] = tuple[Mapping[K, int], Mapping[K, float]]
"""One cell's count per category and its prior weight per category, as `fit_kappa` reads them."""

PRIOR_PSEUDOCOUNT: Final = 0.5
"""Added to each category's count in a cell's prior, so a category the rest of the band lacks
keeps a positive prior mean and the Dirichlet-multinomial likelihood stays finite."""


def fit_kappa[K: Hashable](columns: Sequence[Sequence[Term[K]]]) -> float:
    """The `KAPPA_GRID` value maximising the summed Dirichlet-multinomial log-likelihood.

    A cell's term is the log-probability of its records' category sequence under a
    Dirichlet-multinomial of concentration `kappa` times its prior mean; ties go to the smaller
    `kappa`, so no evidence pools nothing.

    Args:
        columns: Per column, per cell, the cell's count per category and its prior, a weight per
            category that is normalised to a mean.

    Returns:
        The chosen `kappa`.

    Raises:
        ValueError: If a counted category has no positive prior weight.
    """
    grid = np.asarray(KAPPA_GRID)
    total = np.zeros(len(grid))
    for column in columns:
        for counts, prior in column:
            records = sum(counts.values())
            if records == 0:
                continue
            mass = sum(prior.values())
            inner = np.zeros(len(grid))
            for category, count in counts.items():
                share = prior.get(category, 0.0) / mass if mass > 0 else 0.0
                if count and share <= 0:
                    msg = f"category {category!r} holds {count} records and no prior weight"
                    raise ValueError(msg)
                if count:
                    inner += gammaln(grid * share + count) - gammaln(grid * share)
            total += inner - (gammaln(grid + records) - gammaln(grid))
    return float(grid[int(np.argmax(total))])


def row_resample(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each cell's quota as whole rows of pooled donors at `kappa` `GROUP_WEIGHT`.

    Raises:
        BaselineError: If the request is by label, or a cell's banded label group is empty.
    """
    refuse_label(request, "row-resample")
    return _fill(train, request, kappa=float(GROUP_WEIGHT), stream=ROW_RESAMPLE_STREAM, whole=True)


def marginals(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each cell's quota with every column from its own pooled donor at `GROUP_WEIGHT`.

    Raises:
        BaselineError: If the request is by label, or a cell's banded label group is empty.
    """
    refuse_label(request, "marginals")
    return _fill(train, request, kappa=float(GROUP_WEIGHT), stream=MARGINALS_STREAM, whole=False)


def eb_marginals(train: pd.DataFrame, request: Request) -> Baseline:
    """As `marginals`, at the `kappa` `fit_kappa` chooses from the training split.

    The fit reads every drawn column, categorised over its label group (`categorise`), in every
    cell of the quota's labels that holds a record and whose banded label group holds another;
    a cell's prior is the rest of its banded label group's categories, each count raised by
    `PRIOR_PSEUDOCOUNT`.

    Raises:
        BaselineError: If the request is by label, or a cell's banded label group is empty.
    """
    refuse_label(request, "eb-marginals")
    kappa = fit_kappa(_evidence(train, request))
    logger.info("eb holdout=%d fill=%s kappa=%s", *fill_names(), kappa)
    return _fill(train, request, kappa=kappa, stream=EB_MARGINALS_STREAM, whole=False)


def refuse_label(request: Request, method: str) -> None:
    """Refuse a request by label, which names no band, on behalf of `method`.

    Raises:
        BaselineError: If the request is not keyed by cell.
    """
    if request.strata is not Strata.CELL:
        msg = f"{method} needs a request keyed by cell, not by {request.strata}"
        raise BaselineError(msg)


def drawn_columns(request: Request) -> list[str]:
    """The columns a donor gives: every column but `sex` and the label, in frame order."""
    frame = request.frame
    return [name for name in frame.columns if name not in {"sex", frame.label}]


def _evidence(train: pd.DataFrame, request: Request) -> list[list[Term[Hashable]]]:
    """Per drawn column, each informative cell's category counts and prior, for `fit_kappa`."""
    frame = request.frame
    cells = cell_keys(train, edges=frame.edges)
    labels = sorted({int(stratum[-1]) for stratum in request.quota})
    columns: list[list[Term[Hashable]]] = []
    for name in drawn_columns(request):
        terms: list[Term[Hashable]] = []
        for label in labels:
            inside = cells[LABEL] == label
            group = train.loc[inside, name]
            categories = categorise(group, group, ordered=name in frame.numeric)
            terms += _terms(categories, cells.loc[inside], edges=frame.edges)
        columns.append(terms)
    return columns


def _terms(
    categories: pd.Series, cells: pd.DataFrame, *, edges: Sequence[int]
) -> list[Term[Hashable]]:
    """Each cell's counts and prior over one label group's `categories`, on `cells`' index.

    A cell enters when it holds a record and the rest of its banded label group holds another.
    """
    names = list(dict.fromkeys(categories))
    terms: list[Term[Hashable]] = []
    for band in age_band_labels(edges):
        in_band = cells["age_band"] == band
        for sex in SEXES.values():
            own = in_band & (cells["sex"] == sex)
            rest = in_band & ~own
            if not own.any() or not rest.any():
                continue
            held = categories[rest].value_counts()
            total = int(rest.sum()) + PRIOR_PSEUDOCOUNT * len(names)
            prior = {c: (held.get(c, 0) + PRIOR_PSEUDOCOUNT) / total for c in names}
            terms.append((categories[own].value_counts().to_dict(), prior))
    return terms


def _fill(
    train: pd.DataFrame, request: Request, *, kappa: float, stream: int, whole: bool
) -> Baseline:
    """Draw every cell's quota by the pooled rule at `kappa`, one donor per row or per column."""
    frame = request.frame
    cells = cell_keys(train, edges=frame.edges)
    drawn = drawn_columns(request)
    slots = 1 if whole else len(drawn)
    rng = np.random.default_rng([request.seed, stream])
    parts: list[pd.DataFrame] = []
    pools: dict[str, dict[str, int]] = {}
    for stratum, rows in request.quota.items():
        key = "/".join(map(str, stratum))
        own, group = banded_pools(cells, stratum, key)
        pools[key] = {"own": len(own), "banded": len(group)}
        donors = np.array(
            [[pooled_donor(rng, own, group, kappa) for _ in range(slots)] for _ in range(rows)],
            dtype="int64",
        ).reshape(rows, slots)
        if whole:
            part = train.iloc[donors[:, 0]].reset_index(drop=True)
        else:
            part = pd.DataFrame(
                {name: train[name].to_numpy()[donors[:, at]] for at, name in enumerate(drawn)}
            )
        parts.append(cell_rows(part, frame, sex=str(stratum[1]), label=int(stratum[2])))
    added = joined(parts, frame)
    params = {"kappa": kappa, "stream": stream, "pool": pools}
    return Baseline(added=added, weights=unit_weights(train), params=params)


def banded_pools(cells: pd.DataFrame, stratum: Stratum, key: str) -> tuple[np.ndarray, np.ndarray]:
    """The training positions of the cell's own records and of its banded label group.

    Raises:
        BaselineError: If the banded label group is empty.
    """
    band, sex, label = stratum
    banded = (cells["age_band"] == band) & (cells[LABEL] == label)
    group = np.flatnonzero(banded.to_numpy())
    own = np.flatnonzero((banded & (cells["sex"] == sex)).to_numpy())
    logger.info(
        "pool holdout=%d fill=%s key=%s own=%d banded=%d", *fill_names(), key, len(own), len(group)
    )
    if not len(group):
        msg = f"the pooled rule has no training record of label {label} in band {band}: {key}"
        raise BaselineError(msg)
    return own, group


def pooled_donor(rng: np.random.Generator, own: np.ndarray, group: np.ndarray, kappa: float) -> int:
    """One donor's position by the pooled rule: a coin at `cell_weight`, then a pool record."""
    pool = own if rng.random() < cell_weight(len(own), kappa) else group
    return int(pool[int(rng.integers(len(pool)))])
