"""The rows a method adds and the weights it gives: their shape, and the check they must pass.

`Baseline.added` has the evaluation frame's columns and dtypes (`bko.datasets.uci`), so the harness
treats a method's rows like any generator's. Added rows are numbered -1, -2, ... so no `record_id`
of the archive can collide with one.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.errors import OutputContractError
from bko.baselines.request import Request, stratum_keys
from bko.datasets.cells import LABEL
from bko.datasets.uci import CATEGORICAL, CODEBOOK, FEATURES, FRACTIONAL, NUMERIC


FRAME_COLUMNS: Final = (*FEATURES, LABEL)
"""The columns of the training split and of every added frame."""

FRAME_DTYPES: Final[Mapping[str, str]] = {
    name: "float64" if name in FRACTIONAL else "int64" for name in FRAME_COLUMNS
}
"""The evaluation frame's dtypes: whole numbers as int64, `oldpeak` as float64."""

FRACTION_DIGITS: Final = 1
"""The decimal places `oldpeak` has in the archive, which added values keep."""

INDEX_NAME: Final = "record_id"
"""The index name of the evaluation frame and of every added frame."""


@dataclass(frozen=True, slots=True, eq=False)
class Baseline:
    """One method's output for one training split.

    Attributes:
        added: The synthetic rows: `FRAME_COLUMNS` with `FRAME_DTYPES`, indexed -1, -2, ... with
            each stratum's rows together, in the quota's order.
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


def no_rows() -> pd.DataFrame:
    """An added frame with no rows, the evaluation columns and their dtypes."""
    columns = {name: pd.Series(dtype=dtype) for name, dtype in FRAME_DTYPES.items()}
    return number(pd.DataFrame(columns))


def unit_weights(train: pd.DataFrame) -> pd.Series:
    """Weight 1.0 for every training record."""
    return pd.Series(1.0, index=train.index, name="weight")


def training_set(train: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """The rows a classifier fits on: real rows with their weights, then added rows at 1.0.

    The frame has `FRAME_COLUMNS` and `weight`; the real rows keep `baseline.weights`.
    """
    real = train.loc[:, list(FRAME_COLUMNS)].assign(weight=baseline.weights)
    return pd.concat([real, baseline.added.assign(weight=1.0)])


def whole(values: pd.DataFrame) -> pd.DataFrame:
    """Round generated values to the archive's precision and cast them to `FRAME_DTYPES`.

    Whole-number columns round to the nearest integer and `oldpeak` to one decimal, never by
    truncation, which would bias every interpolated value down.

    Raises:
        OutputContractError: If a value is missing or infinite.
    """
    out = values.loc[:, list(FRAME_COLUMNS)].astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        msg = "the output breaks its contract: a value is missing or infinite"
        raise OutputContractError(msg)
    integers = [name for name in FRAME_COLUMNS if name not in FRACTIONAL]
    out[integers] = np.rint(out[integers])
    out[list(FRACTIONAL)] = out[list(FRACTIONAL)].round(FRACTION_DIGITS)
    return out.astype(dict(FRAME_DTYPES))


def derived_seed(seed: int, *parts: object) -> int:
    """A seed in `[0, 2**32)` for one part of a run, from the run's seed and the part's names."""
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def keep(rows: int, of: int, *, seed: int, label: object) -> list[int]:
    """Choose `rows` of `of` positions by sha256 rank, as `bko.datasets.split` ranks records."""

    def rank(position: int) -> bytes:
        return hashlib.sha256(f"{seed}:{label}:{position}".encode()).digest()

    return sorted(sorted(range(of), key=rank)[:rows])


def check(baseline: Baseline, train: pd.DataFrame, request: Request) -> None:
    """Refuse an output that breaks a promise `Baseline` documents.

    Raises:
        OutputContractError: Naming the first broken promise.
    """
    added, weights = baseline.added, baseline.weights
    dtypes = added.dtypes.astype(str).to_dict()
    _require(list(added.columns) == list(FRAME_COLUMNS), f"columns are {list(added.columns)}")
    _require(dtypes == dict(FRAME_DTYPES), f"dtypes are {dtypes}")
    expected = list(range(-1, -len(added) - 1, -1))
    _require(
        added.index.name == INDEX_NAME and added.index.tolist() == expected,
        "the index is not -1, -2, ... under record_id",
    )
    _require(bool(np.isfinite(added.to_numpy(dtype="float64")).all()), "a value is missing")
    for name in CATEGORICAL:
        outside = sorted(set(added[name]) - set(CODEBOOK[name]))
        _require(not outside, f"{name} holds {outside}, outside its codebook")
    labels = sorted(set(added[LABEL]))
    _require(set(labels) <= {0, 1}, f"{LABEL} holds {labels}")
    for name in NUMERIC:
        low, high = train[name].min(), train[name].max()
        _require(bool(added[name].between(low, high).all()), f"{name} leaves [{low}, {high}]")
    for name in FRACTIONAL:
        scaled = added[name].to_numpy() * 10**FRACTION_DIGITS
        _require(bool(np.allclose(scaled, np.rint(scaled))), f"{name} has extra decimals")
    keys = stratum_keys(added, request.strata)
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
