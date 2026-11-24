"""The prior-shift arm: `none`'s fit with each requested subgroup's test logits offset.

`prior_shift` is the arm's runner: it adds no rows and logs, per (age band, sex) subgroup of the
gap request's cells, the subgroup's records per label after the fill. At scoring time
`bko.evaluate.run_scores` solves each model's offset on `out_of_fold` predictions of the training
split and moves the subgroup's test records by it through `shifted`. ADR
`research/prior-shift-comparator` records the estimand and the solver.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from bko.baselines.records import Baseline, no_rows, unit_weights
from bko.baselines.request import Request, Strata
from bko.datasets.cells import cell_keys
from bko.datasets.frames import FrameSpec
from bko.evaluate.classifier import Factory, Model, probabilities
from bko.evaluate.errors import EvaluationError


logger = logging.getLogger("bko.evaluate.run")

EPS: Final = 1e-12
"""Probabilities are clipped to [EPS, 1 - EPS] before their logits are taken."""

BOUND: Final = 64.0
"""The bisection searches offsets in [-BOUND, BOUND]."""

ITERATIONS: Final = 100
"""The bisection's halvings."""

FOLDS: Final = 5
"""The out-of-fold predictions' folds, fewer when a label holds fewer records."""

MIN_FOLDS: Final = 2


def prior_shift(train: pd.DataFrame, request: Request) -> Baseline:
    """No rows, unit weights, and each requested subgroup's records per label after the fill.

    The params are `{"shift": {"<band>/<sex>": [negatives, positives]}}`, one entry per (age
    band, sex) of the request's cells in quota order: the subgroup's training records of each
    label plus the request's rows of it.

    Raises:
        EvaluationError: If the request is not keyed by cell.
    """
    frame = request.frame
    if request.strata is not Strata.CELL:
        msg = f"prior-shift needs a request keyed by cell, not by {request.strata}"
        raise EvaluationError(msg)
    keys = cell_keys(train, edges=frame.edges)
    counts: dict[str, list[int]] = {}
    for (band, sex, label), rows in request.quota.items():
        name = f"{band}/{sex}"
        if name not in counts:
            mine = keys[(keys["age_band"] == band) & (keys["sex"] == sex)]
            counts[name] = [int((mine[frame.label] == code).sum()) for code in (0, 1)]
        counts[name][int(label)] += rows
    return Baseline(added=no_rows(frame), weights=unit_weights(train), params={"shift": counts})


def offset(probabilities: np.ndarray, target: float) -> float:
    """The logit offset at which the mean of the moved `probabilities` equals `target`.

    Bisection on [-BOUND, BOUND] for ITERATIONS halvings, the probabilities clipped to
    [EPS, 1 - EPS]; 0.0 for an empty array.
    """
    if probabilities.size == 0:
        return 0.0
    logits = logit(np.clip(probabilities, EPS, 1 - EPS))
    low, high = -BOUND, BOUND
    for _ in range(ITERATIONS):
        middle = (low + high) / 2
        if float(np.mean(expit(logits + middle))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def out_of_fold(
    training: pd.DataFrame, *, factory: Factory, frame: FrameSpec, model: Model, seed: int
) -> pd.Series:
    """Each training record's P(disease) from a fit on the other folds, on `training`'s index.

    `StratifiedKFold` over `frame`'s label, shuffled at `seed`, with FOLDS folds or the smaller
    label's count when that is fewer; each fold's fit is weighted by `training`'s `weight`.

    Raises:
        EvaluationError: If a label holds fewer than 2 records, or as `probabilities`.
    """
    labels = training[frame.label]
    counts = labels.value_counts()
    folds = min(FOLDS, int(counts.min())) if len(counts) == len(frame.label_codes) else 0
    if folds < MIN_FOLDS:
        msg = f"out-of-fold predictions need {MIN_FOLDS} folds; labels hold {counts.to_dict()}"
        raise EvaluationError(msg)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    parts = [
        probabilities(
            training.iloc[fit_at],
            training.iloc[held_at],
            factory=factory,
            frame=frame,
            model=model,
        )
        for fit_at, held_at in splitter.split(training, labels)
    ]
    return pd.concat(parts).reindex(training.index)


def shifted(
    fitted: pd.Series,
    held_out: pd.Series,
    train: pd.DataFrame,
    test: pd.DataFrame,
    counts: Mapping[str, Sequence[int]],
    *,
    edges: Sequence[int],
    holdout: int,
    model: Model,
) -> pd.Series:
    """`fitted`, `test`'s probabilities, with each subgroup of `counts` moved by its offset.

    A subgroup's offset is `offset` of its training records' `held_out` predictions at its target,
    positives over all of `counts`; a subgroup with no training record gets 0. A record outside
    every subgroup, or in one whose offset is 0, keeps its probability. Each subgroup logs
    `shift holdout=%d model=%s subgroup=%s records=%d target=%.6f offset=%.6f`.
    """
    moved = fitted.copy()
    trained, tested = _subgroups(train, edges), _subgroups(test, edges)
    for subgroup, (negatives, positives) in counts.items():
        mine = (trained == subgroup).to_numpy()
        target = positives / (negatives + positives)
        delta = offset(held_out.reindex(train.index).to_numpy()[mine], target)
        logger.info(
            "shift holdout=%d model=%s subgroup=%s records=%d target=%.6f offset=%.6f",
            holdout,
            model,
            subgroup,
            int(mine.sum()),
            target,
            delta,
        )
        at = (tested == subgroup).to_numpy()
        # Exactly 0 only when no offset applies, so `none`'s probabilities stay byte for byte.
        if delta != 0.0 and at.any():  # ruff: ignore[float-equality-comparison]
            logits = logit(np.clip(fitted.to_numpy()[at], EPS, 1 - EPS))
            moved.iloc[np.flatnonzero(at)] = expit(logits + delta)
    return moved


def _subgroups(records: pd.DataFrame, edges: Sequence[int]) -> pd.Series:
    """Each record's `<band>/<sex>`."""
    keys = cell_keys(records, edges=edges)
    return keys["age_band"].astype(str) + "/" + keys["sex"].astype(str)
