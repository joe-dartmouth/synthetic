"""The consistency checks a candidate record must pass before it joins the augmented dataset.

Each field of a candidate runs one chain, and the first check it fails stops the chain:
`number:<field>`, a finite `int` or `float` (or numpy number), never a `bool`; then its bound,
`code:<field>` from the codebook or `range:<field>` from the mapping table's range constraint; then,
for a numeric field, `precision:<field>`, the archive's decimals. A field fails at most one check,
and no check reads a value an earlier one refused. `fields` refuses a candidate that is not a
mapping, which nothing else then reads, and a mapping naming a key outside `FRAME_COLUMNS`. No
check reads an ontology relationship. ADR `research/record-validation-design` records the checks
and the counting rule.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES
from bko.datasets.uci import (
    CATEGORICAL,
    CODEBOOK,
    FRACTION_DIGITS,
    FRACTIONAL,
    FRAME_COLUMNS,
    FRAME_DTYPES,
)
from bko.ontology.mapping import RANGES
from bko.ontology.schema import Codes


logger = logging.getLogger(__name__)

FIELDS: Final = "fields"
"""The check a candidate fails when it is not a mapping or names a key outside `FRAME_COLUMNS`."""

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

_RANGES: Final = MappingProxyType({row.field: _Range(row.low, row.high) for row in RANGES})


def _chain(name: str) -> tuple[tuple[str, _Rule], ...]:
    if name == LABEL:
        return ((f"code:{name}", Codes(name, LABEL_CODES)),)
    if name in CATEGORICAL:
        return ((f"code:{name}", Codes(name, frozenset(CODEBOOK[name]))),)
    digits = FRACTION_DIGITS if name in FRACTIONAL else 0
    return ((f"range:{name}", _RANGES[name]), (f"precision:{name}", _Precision(digits)))


_CHAINS: Final = MappingProxyType({name: _chain(name) for name in FRAME_COLUMNS})

CHECKS: Final[tuple[str, ...]] = (
    FIELDS,
    *(f"number:{name}" for name in FRAME_COLUMNS),
    *(check for chain in _CHAINS.values() for check, _ in chain),
)
"""Every check's name, in the order `failures` reports them."""


def failures(candidate: object) -> tuple[str, ...]:
    """Return the checks `candidate` fails, in `CHECKS` order; empty when it passes them all."""
    return _read(candidate)[0]


def _read(candidate: object) -> tuple[tuple[str, ...], dict[str, int | float]]:
    if not isinstance(candidate, Mapping):
        return (FIELDS,), {}
    failed: set[str] = set()
    if any(key not in FRAME_COLUMNS for key in candidate):
        failed.add(FIELDS)
    values: dict[str, int | float] = {}
    for name, chain in _CHAINS.items():
        value = _number(candidate.get(name))
        if value is None:
            failed.add(f"number:{name}")
            continue
        broken = next((check for check, rule in chain if not rule.holds(value)), None)
        if broken is None:
            values[name] = value
        else:
            failed.add(broken)
    return tuple(check for check in CHECKS if check in failed), values


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate that failed at least one check.

    Attributes:
        position: Its 0-based position among the candidates given.
        failed: The checks it failed, in `CHECKS` order.
    """

    position: int
    failed: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Validated:
    """The outcome of validating a batch of candidates.

    Attributes:
        accepted: The candidates that passed every check, in input order: `FRAME_COLUMNS` with
            `FRAME_DTYPES`, indexed by position under `CANDIDATE_INDEX`. Call
            `bko.baselines.records.number` before using them as a baseline's added rows.
        rejected: The other candidates, in input order.
    """

    accepted: pd.DataFrame
    rejected: tuple[Rejection, ...]

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
        """How many candidates failed each check, for every check in `CHECKS` order.

        A candidate counts once under each check it failed, so the counts can sum past
        `len(rejected)`; validity is `passed / generated`, never read from these counts.
        """
        counts = Counter(check for rejection in self.rejected for check in rejection.failed)
        return MappingProxyType({check: counts[check] for check in CHECKS})


def validate(candidates: Iterable[object]) -> Validated:
    """Check every candidate and keep the ones that pass, never repairing a value.

    Args:
        candidates: One item per record generation produced, decoded and unfiltered. An item that
            is not a mapping is rejected under `FIELDS`, not dropped, so it counts as generated.

    Returns:
        The accepted frame and the rejections; logs one summary line of counts.

    Raises:
        TypeError: If `candidates` is one record, a string, bytes or a frame, not a batch.
    """
    if isinstance(candidates, str | bytes | Mapping | pd.DataFrame):
        msg = f"validate takes a batch of candidates, not a {type(candidates).__name__}"
        raise TypeError(msg)
    rows: dict[int, dict[str, int | float]] = {}
    rejected: list[Rejection] = []
    for position, candidate in enumerate(candidates):
        failed, values = _read(candidate)
        if failed:
            rejected.append(Rejection(position, failed))
        else:
            rows[position] = values
    index = pd.Index(list(rows), name=CANDIDATE_INDEX, dtype="int64")
    columns = {
        name: pd.Series([row[name] for row in rows.values()], index=index, dtype=FRAME_DTYPES[name])
        for name in FRAME_COLUMNS
    }
    result = Validated(accepted=pd.DataFrame(columns, index=index), rejected=tuple(rejected))
    _log(result)
    return result


def _log(result: Validated) -> None:
    if result.generated == 0:
        logger.warning("validated no candidates, so validity is undefined")
        return
    counts = [f"{check}={count}" for check, count in result.rejections.items() if count]
    logger.info(
        "validated generated=%d passed=%d rejected=%d %s",
        result.generated,
        result.passed,
        len(result.rejected),
        " ".join(counts) or "none",
    )
