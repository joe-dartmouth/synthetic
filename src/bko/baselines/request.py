"""What a baseline is asked for: the strata, the rows each one gains, and the seed.

A stratum is a cell (age band, sex and label, `bko.datasets.cells`) or a label alone. A method
fills exactly the `Request` it is given, so methods given one request meet equal row counts stratum
by stratum (ADR `research/baseline-augmentation-design`).
"""

from __future__ import annotations

import enum
import itertools
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import CELL_COLUMNS, LABEL, SEXES, age_band_labels, cell_keys


BASELINE_SEED: Final = 0
"""The seed every method uses unless an experiment's configuration fixes another."""

SEED_LIMIT: Final = 2**32
"""Seeds run from 0 up to this, exclusive: the range numpy's legacy generator accepts."""

type Stratum = tuple[str | int, ...]
"""A stratum's key: `("<55", "female", 1)` for a cell, `(1,)` for a label."""


class Method(enum.StrEnum):
    """The six comparison methods; the first two are the controls."""

    NONE = "none"
    CLASS_WEIGHT = "class-weight"
    SMOTE = "smote"
    ADASYN = "adasyn"
    CTGAN = "ctgan"
    TVAE = "tvae"

    @property
    def adds_rows(self) -> bool:
        """Whether the method adds rows rather than keeping or reweighting the real ones."""
        return self not in {Method.NONE, Method.CLASS_WEIGHT}


class Strata(enum.StrEnum):
    """What a quota is keyed by."""

    CELL = "cell"
    LABEL = "label"

    @property
    def columns(self) -> tuple[str, ...]:
        """The `cell_keys` columns that name a stratum; each key ends with the label."""
        return CELL_COLUMNS if self is Strata.CELL else (LABEL,)

    def grid(self) -> tuple[Stratum, ...]:
        """Every stratum, in the grid order of `bko.datasets.cells.cell_counts`."""
        if self is Strata.LABEL:
            return ((0,), (1,))
        return tuple(itertools.product(age_band_labels(), SEXES.values(), (0, 1)))


def stratum_keys(frame: pd.DataFrame, strata: Strata) -> list[Stratum]:
    """Return each row's stratum, in row order, computed from the row's own values.

    Raises:
        ContractError: If an age, sex or label is outside what `cell_keys` accepts.
    """
    cells = cell_keys(frame)
    return list(zip(*(cells[column].tolist() for column in strata.columns), strict=True))


@dataclass(frozen=True, slots=True, eq=False)
class Request:
    """The rows a row-adding method adds, stratum by stratum, and the seed it derives from.

    Attributes:
        quota: Rows to add per stratum, held read-only in grid order; only strata that gain rows.
        strata: What the quota is keyed by.
        seed: The seed every random choice in the method derives from.
    """

    quota: Mapping[Stratum, int] = field(default_factory=dict)
    strata: Strata = Strata.CELL
    seed: int = BASELINE_SEED

    def __post_init__(self) -> None:
        """Refuse a malformed quota or seed, and freeze the quota in grid order.

        Raises:
            ValueError: If a quota key is not a stratum of `strata`, a row count is below 1, or the
                seed is outside `[0, SEED_LIMIT)`.
        """
        grid = self.strata.grid()
        for stratum, rows in self.quota.items():
            if stratum not in grid or not _is_int(rows) or rows < 1:
                msg = (
                    f"quota {stratum}: {rows} is not a positive row count "
                    f"for a {self.strata} stratum"
                )
                raise ValueError(msg)
        if not _is_int(self.seed) or not 0 <= self.seed < SEED_LIMIT:
            msg = f"seed {self.seed} is outside [0, {SEED_LIMIT})"
            raise ValueError(msg)
        ordered = {stratum: self.quota[stratum] for stratum in grid if stratum in self.quota}
        object.__setattr__(self, "quota", MappingProxyType(ordered))

    @property
    def rows(self) -> int:
        """The rows the request adds in all."""
        return sum(self.quota.values())


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def water_fill(counts: Mapping[Stratum, int], rows: int) -> dict[Stratum, int]:
    """Give `rows` out one at a time, each to the non-empty stratum holding the fewest so far.

    Args:
        counts: Each stratum's training records, in grid order; ties go to the earlier stratum.
        rows: The rows to give out.

    Returns:
        The rows each stratum gains; strata that gain none are absent.

    Raises:
        ValueError: If `rows` is negative, or rows are asked for and every stratum is empty.
    """
    held = {stratum: n for stratum, n in counts.items() if n > 0}
    if rows < 0 or (rows and not held):
        msg = f"cannot give out {rows} rows over {len(held)} non-empty strata"
        raise ValueError(msg)
    order = {stratum: position for position, stratum in enumerate(held)}
    gained: Counter[Stratum] = Counter()
    for _ in range(rows):
        stratum = min(held, key=lambda s: (held[s] + gained[s], order[s]))
        gained[stratum] += 1
    return {stratum: gained[stratum] for stratum in held if gained[stratum]}


def plan(
    train: pd.DataFrame, rows: int, *, strata: Strata = Strata.CELL, seed: int = BASELINE_SEED
) -> Request:
    """Water-fill `rows` over the strata of `train` into a `Request`."""
    held = Counter(stratum_keys(train, strata))
    counts = {stratum: held[stratum] for stratum in strata.grid()}
    return Request(quota=water_fill(counts, rows), strata=strata, seed=seed)
