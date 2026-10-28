"""The rows a method adds and the weights it gives: their shape, and the check they must pass.

`Baseline.added` has its request's frame's columns and dtypes (`bko.datasets.frames.FrameSpec`,
UCI's by default), so the harness treats a method's rows like any generator's. Added rows are
numbered -1, -2, ... so no `record_id` of the archive can collide with one.

`drops` exposes the per-row rules, so a runner that drops rows uses the rules `check` refuses by;
`assess` also says which prescribed line each kept row of a prescribing call consumed.
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
    """Why an added row is dropped, in the order `drops` tests each reason.

    `check` refuses a row for every reason but `CONCEPTS`, which holds only for a request that
    prescribes each row's concepts (`bko.generate.skeleton`).
    """

    RANGE = "range"
    """A measured value lies outside the training split's minimum and maximum."""
    CELL = "cell"
    """The row's own cell is not one the request asks for, or, in a prescribing call, not the
    call's own cell."""
    CONCEPTS = "concepts"
    """The row's concepts equal no line of its call that an earlier kept row left unconsumed."""
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


@dataclass(frozen=True, slots=True)
class Prescribed:
    """One prescribing call's lines and each row's concepts, as `assess` matches them.

    Attributes:
        cell: The cell the call asked for; a row whose own cell is another is dropped as
            `Drop.CELL`, so a line is consumed only in its own cell.
        lines: Per asked record, in the call's order, the concepts it must show.
        shows: Per row, in order, the concepts it shows.
    """

    cell: Stratum
    lines: tuple[frozenset[str], ...]
    shows: tuple[frozenset[str], ...]


@dataclass(frozen=True, slots=True)
class Assessed:
    """Each row's outcome under `assess`.

    Attributes:
        reasons: Per row, the first reason it is dropped for, or None for a row kept.
        lines: Per row, the position of the line a kept row of a prescribing call consumed, or
            None.
    """

    reasons: tuple[Drop | None, ...]
    lines: tuple[int | None, ...]


def assess(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    request: Request,
    *,
    held: Mapping[Stratum, int] | None = None,
    prescribed: Prescribed | None = None,
) -> Assessed:
    """Each row's first drop reason and, for a prescribing call, the line each kept row consumed.

    Rows are read in order, and a row's cell is the one its own values name under the request's
    frame. A row past its cell's quota counts the rows kept before it, `held` included. A row
    dropped for an earlier reason uses none of the quota. With `prescribed`, a row outside the
    call's own cell is dropped for its cell, and after its cell and before the quota a row is
    matched by concept set: it matches the first line, in the call's order, that equals its
    concepts and no earlier kept row consumed, and it is dropped for its concepts if none does. A
    row kept consumes its match; a row dropped consumes nothing.

    Raises:
        ValueError: If `prescribed` does not hold one concept set per row.
    """
    if prescribed is not None and len(prescribed.shows) != len(rows):
        msg = f"prescribed holds {len(prescribed.shows)} concept sets for {len(rows)} rows"
        raise ValueError(msg)
    frame = request.frame
    inside = within_split(rows, train, frame).all(axis="columns").tolist()
    keys = stratum_keys(rows, request.strata, edges=frame.edges)
    taken = Counter(held or {})
    consumed: set[int] = set()
    reasons: list[Drop | None] = []
    lines: list[int | None] = []
    for row, (fits, key) in enumerate(zip(inside, keys, strict=True)):
        match = None if prescribed is None else _match(prescribed, row, consumed)
        if not fits:
            reason: Drop | None = Drop.RANGE
        elif key not in request.quota or (prescribed is not None and key != prescribed.cell):
            reason = Drop.CELL
        elif prescribed is not None and match is None:
            reason = Drop.CONCEPTS
        elif taken[key] >= request.quota[key]:
            reason = Drop.QUOTA
        else:
            taken[key] += 1
            reason = None
            if match is not None:
                consumed.add(match)
        reasons.append(reason)
        lines.append(match if reason is None else None)
    return Assessed(tuple(reasons), tuple(lines))


def _match(prescribed: Prescribed, row: int, consumed: set[int]) -> int | None:
    """The first line of `prescribed` equal to row `row`'s concepts and not `consumed`."""
    shows = prescribed.shows[row]
    return next(
        (at for at, line in enumerate(prescribed.lines) if at not in consumed and line == shows),
        None,
    )


def drops(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    request: Request,
    *,
    held: Mapping[Stratum, int] | None = None,
    prescribed: Prescribed | None = None,
) -> tuple[Drop | None, ...]:
    """The first reason each row is dropped for, or None for a row kept, as `assess` reads them.

    Raises:
        ValueError: As `assess`.
    """
    return assess(rows, train, request, held=held, prescribed=prescribed).reasons


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
