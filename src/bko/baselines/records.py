"""The rows a method adds and the weights it gives: their shape, and the check they must pass.

`Baseline.added` has its request's frame's columns and dtypes (`bko.datasets.frames.FrameSpec`,
UCI's by default), so the harness treats a method's rows like any generator's. Added rows are
numbered -1, -2, ... so no `record_id` of the archive can collide with one.

`drops` exposes the per-row rules, so a runner that drops rows uses the rules `check` refuses by.
"""

from __future__ import annotations

import enum
import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.errors import OutputContractError
from bko.baselines.request import Request, Stratum, stratum_keys
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME


INDEX_NAME: Final = "record_id"
"""The index name of the evaluation frame and of every added frame."""


class Drop(enum.StrEnum):
    """Why `check` would refuse an added row, in the order `drops` tests each reason."""

    RANGE = "range"
    """A measured value lies outside the training split's minimum and maximum."""
    CELL = "cell"
    """The row's own cell is not one the request asks for."""
    QUOTA = "quota"
    """Its cell already holds every row the request asks of it."""


@dataclass(frozen=True, slots=True, eq=False)
class Baseline:
    """One method's output for one training split.

    Attributes:
        added: The synthetic rows: the request's frame's columns with its dtypes, indexed -1, -2,
            ... with each stratum's rows together, in the quota's order.
        weights: One weight per training record, on the training index; 1.0 except under class
            weighting.
        params: What reproduces the run beyond the commit: parameters and neighbour counts.
    """

    added: pd.DataFrame
    weights: pd.Series
    params: Mapping[str, object]


def number(added: pd.DataFrame) -> pd.DataFrame:
    """Index `added` -1, -2, ... in row order, under the name `record_id`."""
    out = added.reset_index(drop=True)
    out.index = pd.Index(range(-1, -len(out) - 1, -1), name=INDEX_NAME, dtype="int64")
    return out


def no_rows(frame: FrameSpec = UCI_FRAME) -> pd.DataFrame:
    """An added frame with no rows, `frame`'s columns and their dtypes."""
    columns = {name: pd.Series(dtype=dtype) for name, dtype in frame.dtypes.items()}
    return number(pd.DataFrame(columns))


def unit_weights(train: pd.DataFrame) -> pd.Series:
    """Weight 1.0 for every training record."""
    return pd.Series(1.0, index=train.index, name="weight")


def training_set(
    train: pd.DataFrame, baseline: Baseline, *, frame: FrameSpec = UCI_FRAME
) -> pd.DataFrame:
    """The rows a classifier fits on: real rows with their weights, then added rows at 1.0.

    The result has `frame`'s columns and `weight`; the real rows keep `baseline.weights`.
    """
    real = train.loc[:, list(frame.columns)].assign(weight=baseline.weights)
    return pd.concat([real, baseline.added.assign(weight=1.0)])


def whole(values: pd.DataFrame, frame: FrameSpec = UCI_FRAME) -> pd.DataFrame:
    """Round generated values to `frame`'s precision and cast them to its dtypes.

    Whole-number columns round to the nearest integer and fractional ones (UCI's `oldpeak`) to
    the frame's decimals, never by truncation, which would bias every interpolated value down.

    Raises:
        OutputContractError: If a value is missing or infinite.
    """
    out = values.loc[:, list(frame.columns)].astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        msg = "the output breaks its contract: a value is missing or infinite"
        raise OutputContractError(msg)
    integers = [name for name in frame.columns if name not in frame.fractional]
    out[integers] = np.rint(out[integers])
    out[list(frame.fractional)] = out[list(frame.fractional)].round(frame.fraction_digits)
    return out.astype(dict(frame.dtypes))


def derived_seed(seed: int, *parts: object) -> int:
    """A seed in `[0, 2**32)` for one part of a run, from the run's seed and the part's names."""
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def keep(rows: int, of: int, *, seed: int, label: object) -> list[int]:
    """Choose `rows` of `of` positions by sha256 rank, as `bko.datasets.split` ranks records."""

    def rank(position: int) -> bytes:
        return hashlib.sha256(f"{seed}:{label}:{position}".encode()).digest()

    return sorted(sorted(range(of), key=rank)[:rows])


def within_split(
    rows: pd.DataFrame, train: pd.DataFrame, frame: FrameSpec = UCI_FRAME
) -> pd.DataFrame:
    """Whether each row's value of each of `frame`'s numeric columns lies in the split's range."""
    return pd.DataFrame(
        {name: rows[name].between(train[name].min(), train[name].max()) for name in frame.numeric},
        index=rows.index,
    )


def drops(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    request: Request,
    *,
    held: Mapping[Stratum, int] | None = None,
) -> tuple[Drop | None, ...]:
    """The first reason `check` would refuse each row for, or None for a row it would keep.

    Rows are read in order, and a row's cell is the one its own values name under the request's
    frame. A row past its cell's quota counts the rows kept before it, `held` included. A row
    dropped for an earlier reason uses none of the quota.
    """
    frame = request.frame
    inside = within_split(rows, train, frame).all(axis="columns").tolist()
    keys = stratum_keys(rows, request.strata, edges=frame.edges)
    taken = Counter(held or {})
    reasons: list[Drop | None] = []
    for fits, key in zip(inside, keys, strict=True):
        if not fits:
            reasons.append(Drop.RANGE)
        elif key not in request.quota:
            reasons.append(Drop.CELL)
        elif taken[key] >= request.quota[key]:
            reasons.append(Drop.QUOTA)
        else:
            taken[key] += 1
            reasons.append(None)
    return tuple(reasons)


def check(baseline: Baseline, train: pd.DataFrame, request: Request) -> None:
    """Refuse an output that breaks a promise `Baseline` documents, under the request's frame.

    Raises:
        OutputContractError: Naming the first broken promise.
    """
    frame = request.frame
    added, weights = baseline.added, baseline.weights
    dtypes = added.dtypes.astype(str).to_dict()
    _require(list(added.columns) == list(frame.columns), f"columns are {list(added.columns)}")
    _require(dtypes == dict(frame.dtypes), f"dtypes are {dtypes}")
    expected = list(range(-1, -len(added) - 1, -1))
    _require(
        added.index.name == INDEX_NAME and added.index.tolist() == expected,
        "the index is not -1, -2, ... under record_id",
    )
    _require(bool(np.isfinite(added.to_numpy(dtype="float64")).all()), "a value is missing")
    for name in frame.categorical:
        outside = sorted(set(added[name]) - set(frame.codebook[name]))
        _require(not outside, f"{name} holds {outside}, outside its codebook")
    labels = sorted(set(added[frame.label]))
    _require(set(labels) <= set(frame.label_codes), f"{frame.label} holds {labels}")
    inside = within_split(added, train, frame)
    for name in frame.numeric:
        low, high = train[name].min(), train[name].max()
        _require(bool(inside[name].all()), f"{name} leaves [{low}, {high}]")
    for name in frame.fractional:
        scaled = added[name].to_numpy() * 10**frame.fraction_digits
        _require(bool(np.allclose(scaled, np.rint(scaled))), f"{name} has extra decimals")
    keys = stratum_keys(added, request.strata, edges=frame.edges)
    _require(Counter(keys) == Counter(request.quota), f"rows per stratum are {dict(Counter(keys))}")
    in_order = [stratum for stratum, rows in request.quota.items() for _ in range(rows)]
    _require(keys == in_order, "the strata are not together in the quota's order")
    _require(weights.index.equals(train.index), "the weights are not on the training index")
    finite = bool(np.isfinite(weights.to_numpy()).all())
    _require(finite and bool((weights > 0).all()), "a weight is not a positive number")


def _require(holds: bool, problem: str) -> None:
    if not holds:
        msg = f"the output breaks its contract: {problem}"
        raise OutputContractError(msg)
