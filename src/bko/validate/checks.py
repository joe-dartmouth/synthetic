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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from bko.datasets.cells import LABEL, LABEL_CODES
from bko.datasets.uci import CATEGORICAL, CODEBOOK, FRACTION_DIGITS, FRACTIONAL, FRAME_COLUMNS
from bko.ontology.mapping import RANGES
from bko.ontology.schema import Codes


FIELDS: Final = "fields"
"""The check a candidate fails when it is not a mapping or names a key outside `FRAME_COLUMNS`."""


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
