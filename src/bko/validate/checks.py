"""The consistency checks a candidate record must pass before it joins the augmented dataset.

Each field of a candidate runs one chain, and the first check it fails stops the chain:
`number:<field>`, a finite `int`, `float` or numpy integer, never a `bool`; then its bound,
`code:<field>` from the frame's codebook or label codes, or `range:<field>` from its range
constraint; then, for a numeric field, `precision:<field>`, the frame's decimals. A field fails at
most one check, and no check reads a value an earlier one refused. `fields` refuses a candidate
that is not a mapping, which nothing else then reads, and a mapping naming a key outside the
frame's columns. The frame is a required keyword, so no caller checks against UCI's by
default. No check reads an ontology relationship. ADR `research/record-validation-design` records
the checks and the counting rule.
"""

from __future__ import annotations

import functools
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.ontology.schema import Codes


logger = logging.getLogger(__name__)

FIELDS: Final = "fields"
"""The check a candidate fails when it is not a mapping or names a key outside its frame."""

CANDIDATE_INDEX: Final = "candidate"
"""The accepted frame's index name: each row's 0-based position among the candidates given."""


@dataclass(frozen=True, slots=True)
class _Range:
    low: float
    high: float

    def holds(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True, slots=True)
class _Precision:
    digits: int

    def holds(self, value: float) -> bool:
        return round(value, self.digits) == value


type _Rule = Codes | _Range | _Precision
type _Chain = tuple[tuple[str, _Rule], ...]


def _chain(name: str, frame: FrameSpec) -> _Chain:
    if name == frame.label:
        return ((f"code:{name}", Codes(name, frozenset(frame.label_codes))),)
    if name in frame.categorical:
        return ((f"code:{name}", Codes(name, frozenset(frame.codebook[name]))),)
    bound = next(row for row in frame.ranges if row.field == name)
    digits = frame.fraction_digits if name in frame.fractional else 0
    return (
        (f"range:{name}", _Range(bound.low, bound.high)),
        (f"precision:{name}", _Precision(digits)),
    )


@functools.cache
def _chains(frame: FrameSpec) -> Mapping[str, _Chain]:
    return MappingProxyType({name: _chain(name, frame) for name in frame.columns})


@functools.cache
def checks(frame: FrameSpec) -> tuple[str, ...]:
    """Return every check's name under `frame`, in the order `failures` reports them."""
    return (
        FIELDS,
        *(f"number:{name}" for name in frame.columns),
        *(check for chain in _chains(frame).values() for check, _ in chain),
    )


CHECKS: Final[tuple[str, ...]] = checks(UCI_FRAME)
"""Every check's name under `UCI_FRAME`, in the order `failures` reports them."""


def failures(candidate: object, *, frame: FrameSpec) -> tuple[str, ...]:
    """Return the checks `candidate` fails, in `checks(frame)` order; empty when it passes all."""
    return _read(candidate, frame)[0]


def _read(candidate: object, frame: FrameSpec) -> tuple[tuple[str, ...], dict[str, int | float]]:
    if not isinstance(candidate, Mapping):
        return (FIELDS,), {}
    failed: set[str] = set()
    columns = frame.columns
    if any(key not in columns for key in candidate):
        failed.add(FIELDS)
    values: dict[str, int | float] = {}
    for name, chain in _chains(frame).items():
        value = _number(candidate.get(name))
        if value is None:
            failed.add(f"number:{name}")
            continue
        broken = next((check for check, rule in chain if not rule.holds(value)), None)
        if broken is None:
            values[name] = value
        else:
            failed.add(broken)
    return tuple(check for check in checks(frame) if check in failed), values


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate that failed at least one check.

    Attributes:
        position: Its 0-based position among the candidates given.
        failed: The checks it failed, in `checks(frame)` order.
    """

    position: int
    failed: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Validated:
    """The outcome of validating a batch of candidates.

    Attributes:
        accepted: The candidates that passed every check, in input order: the frame's columns
            with its dtypes, indexed by position under `CANDIDATE_INDEX`. Call
            `bko.baselines.records.number` before using them as a baseline's added rows.
        rejected: The other candidates, in input order.
        frame: The frame they were checked against.
    """

    accepted: pd.DataFrame
    rejected: tuple[Rejection, ...]
    frame: FrameSpec

    @property
    def generated(self) -> int:
        """How many candidates were given: the validity criterion's denominator."""
        return len(self.accepted) + len(self.rejected)

    @property
    def passed(self) -> int:
        """How many passed every check: the validity criterion's numerator."""
        return len(self.accepted)

    @property
    def rejections(self) -> Mapping[str, int]:
        """How many candidates failed each check, for every check in `checks(frame)` order.

        A candidate counts once under each check it failed, so the counts can sum past
        `len(rejected)`; validity is `passed / generated`, never read from these counts.
        """
        counts = Counter(check for rejection in self.rejected for check in rejection.failed)
        return MappingProxyType({check: counts[check] for check in checks(self.frame)})


def validate(candidates: Iterable[object], *, frame: FrameSpec) -> Validated:
    """Check every candidate against `frame` and keep the ones that pass, never repairing a value.

    Args:
        candidates: One item per record generation produced, decoded and unfiltered, plus one
            placeholder that is not a mapping per record a call that returned a response asked
            for and did not decode. An item that is not a mapping is rejected under `FIELDS`,
            not dropped, so it counts as generated.
        frame: The frame whose columns, codes, ranges and decimals a candidate must meet.

    Returns:
        The accepted frame and the rejections; logs one summary line of counts, or a warning when
        there are no candidates.

    Raises:
        TypeError: If `candidates` is one record, a string, a bytes-like object or a frame, not a
            batch.
    """
    if isinstance(candidates, str | bytes | bytearray | memoryview | Mapping | pd.DataFrame):
        msg = f"validate takes a batch of candidates, not a {type(candidates).__name__}"
        raise TypeError(msg)
    rows: dict[int, dict[str, int | float]] = {}
    rejected: list[Rejection] = []
    for position, candidate in enumerate(candidates):
        failed, values = _read(candidate, frame)
        if failed:
            rejected.append(Rejection(position, failed))
        else:
            rows[position] = values
    index = pd.Index(list(rows), name=CANDIDATE_INDEX, dtype="int64")
    dtypes = frame.dtypes
    columns = {
        name: pd.Series([row[name] for row in rows.values()], index=index, dtype=dtypes[name])
        for name in frame.columns
    }
    accepted = pd.DataFrame(columns, index=index)
    result = Validated(accepted=accepted, rejected=tuple(rejected), frame=frame)
    _log(result)
    return result


def _log(result: Validated) -> None:
    if result.generated == 0:
        logger.warning(
            "validated no candidates frame=%s, so validity is undefined", result.frame.name
        )
        return
    counts = [f"{check}={count}" for check, count in result.rejections.items() if count]
    logger.info(
        "validated generated=%d passed=%d rejected=%d %s frame=%s",
        result.generated,
        result.passed,
        len(result.rejected),
        " ".join(counts) or "none",
        result.frame.name,
    )
