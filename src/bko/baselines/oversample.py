"""SMOTE and ADASYN from imbalanced-learn, one call per stratum that gains rows.

Each call oversamples one stratum against the rest of the training split, so its neighbours, and
every row it interpolates, stay inside the stratum. Measured predictors are z-scored with the
training split's mean and standard deviation for the neighbour search, then restored and rounded
(`records.whole`). SMOTE runs as SMOTE-NC, which takes each coded predictor from the neighbours.
ADASYN has no such variant: the coded predictors are one-hot encoded and each block decoded to its
largest entry, the nearer parent's code. ADASYN rounds each record's share, so it is asked for
ceil(records / 2) extra rows and the quota is kept by sha256 rank. The fields are the request's
frame's. ADR `research/baseline-augmentation-design` records the choices.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTENC

from bko.baselines.errors import BaselineError
from bko.baselines.records import Baseline, derived_seed, keep, number, unit_weights, whole
from bko.baselines.request import Request, stratum_keys
from bko.datasets.frames import FrameSpec


NEIGHBORS: Final = 5
"""SMOTE's and ADASYN's neighbour count, imbalanced-learn's default; n records use at most n - 1."""

ONE_HOT_SCALE: Final = 1 / math.sqrt(2)
"""Scales ADASYN's one-hot entries so one differing code adds a distance of 1, one standard
deviation of a z-scored predictor."""


@dataclass(frozen=True, slots=True, eq=False)
class _Scaled:
    numeric: np.ndarray
    mean: pd.Series
    scale: pd.Series


type _Sampler = Callable[
    [pd.DataFrame, _Scaled, np.ndarray, int, int, int, FrameSpec], pd.DataFrame
]


def smote(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each stratum's quota with SMOTE-NC."""
    return _oversample(train, request, _smote_stratum, method="smote")


def adasyn(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each stratum's quota with ADASYN on one-hot coded predictors."""
    return _oversample(train, request, _adasyn_stratum, method="adasyn")


def _oversample(
    train: pd.DataFrame, request: Request, sample: _Sampler, *, method: str
) -> Baseline:
    frame = request.frame
    keys = stratum_keys(train, request.strata, edges=frame.edges)
    scaled = _scale(train, frame)
    parts: list[pd.DataFrame] = []
    neighbors: dict[str, int] = {}
    for stratum, rows in request.quota.items():
        inside = np.fromiter((key == stratum for key in keys), dtype=bool, count=len(keys))
        size = int(inside.sum())
        if size < 2:
            msg = f"{method} needs at least 2 training records in stratum {stratum}; it has {size}"
            raise BaselineError(msg)
        neighbors[str(stratum)] = k = min(NEIGHBORS, size - 1)
        seed = derived_seed(request.seed, method, *stratum)
        values = sample(train, scaled, inside, rows, k, seed, frame)
        values[frame.label] = stratum[-1]
        parts.append(whole(values, frame))
    added = number(pd.concat(parts))
    return Baseline(added=added, weights=unit_weights(train), params={"neighbors": neighbors})


def _smote_stratum(
    train: pd.DataFrame,
    scaled: _Scaled,
    inside: np.ndarray,
    rows: int,
    k: int,
    seed: int,
    frame: FrameSpec,
) -> pd.DataFrame:
    numeric, categorical = len(frame.numeric), list(frame.categorical)
    x = np.column_stack([scaled.numeric, train[categorical].to_numpy(dtype="float64")])
    coded = list(range(numeric, numeric + len(categorical)))
    sampler = SMOTENC(
        categorical_features=coded,
        sampling_strategy={1: int(inside.sum()) + rows},
        k_neighbors=k,
        random_state=seed,
    )
    try:
        resampled, _ = sampler.fit_resample(x, inside.astype("int64"))
    except (RuntimeError, ValueError) as exc:
        msg = f"smote cannot fill a stratum of {int(inside.sum())} records: {exc}"
        raise BaselineError(msg) from exc
    new = resampled[len(x) :]
    return _restore(scaled, new[:, :numeric], new[:, numeric:], frame)


def _adasyn_stratum(
    train: pd.DataFrame,
    scaled: _Scaled,
    inside: np.ndarray,
    rows: int,
    k: int,
    seed: int,
    frame: FrameSpec,
) -> pd.DataFrame:
    codes = {name: sorted(frame.codebook[name]) for name in frame.categorical}
    blocks = [np.equal.outer(train[name].to_numpy(), codes[name]) for name in frame.categorical]
    x = np.column_stack([scaled.numeric, *(block * ONE_HOT_SCALE for block in blocks)])
    size = int(inside.sum())
    sampler = ADASYN(
        sampling_strategy={1: size + rows + math.ceil(size / 2)}, n_neighbors=k, random_state=seed
    )
    try:
        resampled, _ = sampler.fit_resample(x, inside.astype("int64"))
    except (RuntimeError, ValueError) as exc:
        msg = f"adasyn cannot fill a stratum of {size} records: {exc}"
        raise BaselineError(msg) from exc
    new = resampled[len(x) :]
    new = new[keep(rows, len(new), seed=seed, label="adasyn")]
    decoded = np.empty((len(new), len(frame.categorical)))
    start = len(frame.numeric)
    for column, name in enumerate(frame.categorical):
        width = len(codes[name])
        picks = new[:, start : start + width].argmax(axis=1)
        decoded[:, column] = np.asarray(codes[name])[picks]
        start += width
    return _restore(scaled, new[:, : len(frame.numeric)], decoded, frame)


def _scale(train: pd.DataFrame, frame: FrameSpec) -> _Scaled:
    numeric = train[list(frame.numeric)].astype("float64")
    mean = numeric.mean()
    scale = numeric.std(ddof=0).replace(0.0, 1.0)
    return _Scaled(numeric=((numeric - mean) / scale).to_numpy(), mean=mean, scale=scale)


def _restore(
    scaled: _Scaled, numeric: np.ndarray, coded: np.ndarray, frame: FrameSpec
) -> pd.DataFrame:
    measured = pd.DataFrame(numeric, columns=list(frame.numeric)) * scaled.scale + scaled.mean
    categorical = pd.DataFrame(coded, columns=list(frame.categorical))
    return pd.concat([measured, categorical], axis=1).loc[:, list(frame.features)]
