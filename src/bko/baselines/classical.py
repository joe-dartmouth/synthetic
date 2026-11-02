"""The classical synthesizers: SDV's Gaussian copula and sequential CART after synthpop.

Each draws whole records unconditioned and keeps a stratum's first matching rows until its quota
is met (`bko.baselines.synthesizers.fill`), as CTGAN and TVAE do. ADR
`research/baseline-augmentation-design` records the settings and why.

The copula is SDV's `GaussianCopulaSynthesizer` at its defaults. SDV seeds its sampler with a fixed
seed of its own (`FIXED_RNG_SEED`) whatever the global generators hold, so the request's seed moves
nothing: equal training rows give equal rows at every seed.

Sequential CART follows synthpop's `syn.cart` (CRAN 1.9-3) but for one departure: scikit-learn
trees split on one threshold, so a code with 3 or more values enters as one-hot indicators where
rpart splits on subsets of its values (ADR `research/baseline-augmentation-design`). The columns
are visited in the frame's order, the first drawn from its training values, and each later one from
a tree fit on the columns before it, copying the value of a uniformly chosen training row in the
new row's leaf. The trees use rpart's defaults as synthpop sets them; scikit-learn grows them and
draws nothing at random once its seed is fixed.

Each method logs `fit method=<name> seconds=<s>` as CTGAN and TVAE do: the copula's covers its
`fit`, CART's the fit of every column's tree and the leaf each training row lands in. Seconds are
wall time, a liveness log and not a cost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from bko.baselines.records import Baseline, derived_seed, unit_weights
from bko.baselines.request import Request
from bko.baselines.synthesizers import fill, metadata
from bko.datasets.frames import FrameSpec


logger = logging.getLogger(__name__)

CART_MIN_LEAF: Final = 5
"""synthpop's `minbucket`: the fewest training rows a leaf holds."""

CART_MIN_SPLIT: Final = 15
"""rpart's `minsplit`, three times `minbucket`: the fewest rows a node needs to split."""

CART_MAX_DEPTH: Final = 30
"""rpart's `maxdepth`."""

CART_COMPLEXITY: Final = 1e-8
"""synthpop's `cp`: a split must cut the impurity by this share of the root's.

`_tree` multiplies it by the column's own root impurity (the Gini index of a coded column, the
variance of a measured one) for scikit-learn's `min_impurity_decrease`, as rpart scales `cp` by
the root's risk."""


def gaussian_copula(train: pd.DataFrame, request: Request) -> Baseline:
    """SDV's Gaussian copula at its defaults, fit on `train`, rows kept by stratum.

    The request's seed reaches nothing: SDV fixes the copula's sampling seed itself.

    Raises:
        BaselineError: As `fill`, if the copula's draws cannot meet the quota; its shortfall holds
            no baseline, so the harness refuses it rather than filling partially.
    """
    frame = request.frame
    tables = Metadata.load_from_dict(dict(metadata(frame)))
    synthesizer = GaussianCopulaSynthesizer(tables)
    start = time.perf_counter()
    synthesizer.fit(train.loc[:, list(frame.columns)])
    logger.info("fit method=gaussian-copula seconds=%.1f", time.perf_counter() - start)
    added = fill(lambda rows: synthesizer.sample(num_rows=rows), request, method="gaussian-copula")
    return Baseline(added=added, weights=unit_weights(train), params=synthesizer.get_parameters())


def cart(train: pd.DataFrame, request: Request) -> Baseline:
    """Sequential CART synthesis after synthpop, quota met by rejection.

    Trees are fit once per column on the training rows; each batch then draws the first column
    from its training values and every later column from a same-leaf training donor. Every draw
    reads `default_rng(derived_seed(seed, "cart"))`, so the seed alone decides the rows, and no
    global generator is read or moved.

    Raises:
        BaselineError: As `fill`, if the draws cannot meet the quota; its shortfall holds no
            baseline, so the harness refuses it rather than filling partially.
    """
    frame = request.frame
    columns = list(frame.columns)
    rows = train.loc[:, columns].reset_index(drop=True)
    start = time.perf_counter()
    trees = [_tree(rows, columns, at, frame, seed=request.seed) for at in range(1, len(columns))]
    leaves = [tree.apply(_predictors(rows, columns[:at], frame)) for at, tree in trees]
    logger.info("fit method=cart seconds=%.1f", time.perf_counter() - start)
    rng = np.random.default_rng(derived_seed(request.seed, "cart"))

    def draw(size: int) -> pd.DataFrame:
        out = pd.DataFrame(index=range(size))
        out[columns[0]] = rng.choice(rows[columns[0]].to_numpy(), size=size)
        for (at, tree), trained in zip(trees, leaves, strict=True):
            column = columns[at]
            new = tree.apply(_predictors(out, columns[:at], frame))
            out[column] = _donated(new, trained, rows[column].to_numpy(), rng)
        return out.astype(dict(frame.dtypes))

    added = fill(draw, request, method="cart")
    params = {
        "seed": request.seed,
        "visit": columns,
        "min_samples_leaf": CART_MIN_LEAF,
        "min_samples_split": CART_MIN_SPLIT,
        "max_depth": CART_MAX_DEPTH,
        "complexity": CART_COMPLEXITY,
    }
    return Baseline(added=added, weights=unit_weights(train), params=params)


def _coded(column: str, frame: FrameSpec) -> tuple[int, ...] | None:
    """The codes of a coded column of `frame`, the label included; None for a measured one."""
    if column == frame.label:
        return tuple(frame.label_codes)
    if column in frame.categorical:
        return tuple(frame.codebook[column])
    return None


def _predictors(rows: pd.DataFrame, columns: Sequence[str], frame: FrameSpec) -> np.ndarray:
    """`rows`' `columns` as tree predictors: a code with 3 or more values one-hot, others as is."""
    parts: list[np.ndarray] = []
    for column in columns:
        values = rows[column].to_numpy()
        codes = _coded(column, frame)
        if codes is not None and len(codes) >= 3:
            parts.append(np.equal.outer(values, np.asarray(codes)).astype(float))
        else:
            parts.append(values.astype(float).reshape(-1, 1))
    return np.hstack(parts)


def _tree(
    rows: pd.DataFrame, columns: Sequence[str], at: int, frame: FrameSpec, *, seed: int
) -> tuple[int, DecisionTreeClassifier | DecisionTreeRegressor]:
    """Column `at`'s tree, fit on the columns before it, with synthpop's settings."""
    column = columns[at]
    target = rows[column].to_numpy()
    coded = _coded(column, frame) is not None
    if coded:
        _, counts = np.unique(target, return_counts=True)
        shares = counts / counts.sum()
        root = 1.0 - float(np.sum(shares**2))
    else:
        root = float(np.var(target))
    settings = {
        "min_samples_leaf": CART_MIN_LEAF,
        "min_samples_split": CART_MIN_SPLIT,
        "max_depth": CART_MAX_DEPTH,
        "min_impurity_decrease": CART_COMPLEXITY * root,
        "random_state": derived_seed(seed, "cart", column),
    }
    tree = DecisionTreeClassifier(**settings) if coded else DecisionTreeRegressor(**settings)
    tree.fit(_predictors(rows, columns[:at], frame), target)
    return at, tree


def _donated(
    new: np.ndarray, trained: np.ndarray, values: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For each new row's leaf, the value of a uniformly chosen training row in that leaf."""
    out = np.empty(len(new), dtype=values.dtype)
    order = np.argsort(trained, kind="stable")
    sorted_leaves = trained[order]
    for leaf in np.unique(new):
        at = new == leaf
        start, stop = np.searchsorted(sorted_leaves, [leaf, leaf + 1])
        donors = order[start:stop]
        out[at] = values[donors[rng.integers(len(donors), size=int(at.sum()))]]
    return out
