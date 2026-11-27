"""How many rows a flagged cell is asked for under each quota rule but the fixed fill.

The harness asks each flagged cell for `fill_to - count` rows (`QuotaRule.FILL_TO`). The other
rules derive a count from the cell itself: `coverage`, the smallest count at which every concept
of fused share at least `SHARE_FLOOR` appears at least once with probability at least
1 - `MISS_CHANCE`, by a union bound; `expected_size`, the cell's size were its label split the one
its neighbours along the second cell axis hold; and `recommended`, the larger of the two less what
the cell holds. ADR `research/share-based-quota` records the rule. This module imports nothing from
`bko.generate` or `bko.evaluate`, so the gap stage owns the counts.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import counts_by_cell
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, observe_records, record_cells
from bko.gaps.shares import shares
from bko.ontology.schema import FieldMapping


class QuotaRule(enum.StrEnum):
    """How many rows each flagged cell of a gap request is asked for."""

    FILL_TO = "fill-to"
    COVERAGE = "coverage"
    COVERAGE_2 = "coverage-x2"
    COVERAGE_4 = "coverage-x4"
    COVERAGE_8 = "coverage-x8"
    EXPECTED = "expected-size"
    RECOMMENDED = "recommended"


SHARE_FLOOR: Final = 0.02
"""The least fused share at which a concept must appear in a covered cell."""

MISS_CHANCE: Final = 0.05
"""The most a covered cell's chance of missing any concept at or above `SHARE_FLOOR` may be."""

MULTIPLES: Final[Mapping[QuotaRule, int]] = MappingProxyType(
    {
        QuotaRule.COVERAGE: 1,
        QuotaRule.COVERAGE_2: 2,
        QuotaRule.COVERAGE_4: 4,
        QuotaRule.COVERAGE_8: 8,
    }
)
"""Each coverage rule's multiple of `coverage`."""


@dataclass(frozen=True, slots=True)
class Ask:
    """One flagged cell's counts under a quota rule.

    Attributes:
        count: The training records the cell holds.
        coverage: `coverage` of the cell's fused shares.
        expected: `expected_size` of the cell.
        asked: The rows the rule asks for; 0 leaves the cell out of the request.
    """

    count: int
    coverage: int
    expected: int
    asked: int


DENOMINATOR: Final = 10**9
"""The largest denominator a share is read at: each share is the float nearest a ratio of counts."""


def _exact(value: float) -> Fraction:
    """`value` as the ratio of denominator at most `DENOMINATOR` nearest it."""
    return Fraction(value).limit_denominator(DENOMINATOR)


def coverage(share: Mapping[str, float]) -> int:
    """The smallest n at which every concept of share at least `SHARE_FLOOR` appears, w.p. 0.95.

    By a union bound: the least n at which the sum over those concepts of (1 - p)^n is at most
    `MISS_CHANCE`; 0 when no concept qualifies. Each share, the floor and the miss chance are read
    as exact fractions (`_exact`), so a share one ulp off a ratio of counts reads as that ratio and
    a sum that meets `MISS_CHANCE` exactly stops.

    Raises:
        ValueError: If a share is NaN.
    """
    if any(math.isnan(p) for p in share.values()):
        msg = f"a share is NaN: {dict(share)}"
        raise ValueError(msg)
    floor, chance = _exact(SHARE_FLOOR), _exact(MISS_CHANCE)
    misses = [1 - exact for p in share.values() if (exact := _exact(p)) >= floor]
    if not misses:
        return 0
    n, powers = 0, [Fraction(1)] * len(misses)
    while sum(powers) > chance:
        n += 1
        powers = [power * miss for power, miss in zip(powers, misses, strict=True)]
    return n


def expected_size(cell: Cell, counts: Mapping[Cell, int]) -> int:
    """The cell's size were its label split its neighbours' along the second cell axis.

    For cell (a, b, y): n(a, b, not y) (N1 + 1/2) / (N0 + 1/2), N1 and N0 the counts of
    (a, b', y) and (a, b', not y) over every other value b' of the second axis, read by position.
    When both are 0, N1 and N0 are every cell's counts of y and of not y. Exact, rounded half up.
    """
    first, second, label = cell[0], cell[1], cell[2]
    others = [key for key in counts if key[0] == first and key[1] != second]
    n1 = sum(counts[key] for key in others if key[2] == label)
    n0 = sum(counts[key] for key in others if key[2] != label)
    if n1 == n0 == 0:
        n1 = sum(count for key, count in counts.items() if key[2] == label)
        n0 = sum(count for key, count in counts.items() if key[2] != label)
    opposite = sum(
        count
        for key, count in counts.items()
        if (key[0], key[1]) == (first, second) and key[2] != label
    )
    size = opposite * (n1 + Fraction(1, 2)) / (n0 + Fraction(1, 2))
    return math.floor(size + Fraction(1, 2))


def recommended(count: int, covered: int, expected: int) -> int:
    """The rows a cell holding `count` records needs to reach both counts: never below 0."""
    return max(0, max(covered, expected) - count)


def fused_shares(
    train: pd.DataFrame,
    cells: Sequence[Cell],
    *,
    name: str,
    rows: Sequence[FieldMapping],
    frame: FrameSpec,
) -> dict[Cell, dict[str, float]]:
    """Each of `cells`' fused concept shares, read as the direct sampler reads them.

    `bko.gaps.shares.shares` of the cell's records against its label group's, observed under
    `rows`.

    Raises:
        GapError: As `bko.gaps.observe.record_cells` and `observe_records`.
    """
    observed = record_cells(train, rows, name=name, edges=frame.edges).each
    groups = {
        label: observe_records(
            train.loc[train[frame.label].eq(label)], rows, name=f"{name}/{frame.label}={label}"
        )
        for label in sorted({int(cell[2]) for cell in cells})
    }
    return {cell: shares(observed[cell], groups[int(cell[2])]) for cell in cells}


def asks(
    train: pd.DataFrame,
    cells: Sequence[Cell],
    rule: QuotaRule,
    *,
    name: str,
    rows: Sequence[FieldMapping],
    frame: FrameSpec,
    counts: Mapping[Cell, int] | None = None,
) -> dict[Cell, Ask]:
    """Each of `cells`' counts under `rule`, a rule other than `QuotaRule.FILL_TO`.

    The fused shares are `fused_shares`'. `counts` is every cell's training count
    (`bko.datasets.cells.counts_by_cell` of `train`), read from `train` when None; no cell reads
    no share.

    Raises:
        ValueError: If `rule` is `QuotaRule.FILL_TO`, whose count is the experiment's `fill_to`,
            or no rule this module knows.
        GapError: As `bko.gaps.observe.record_cells` and `observe_records`.
    """
    if rule is QuotaRule.FILL_TO:
        msg = f"the {rule} rule asks fill_to less the count, which this module does not hold"
        raise ValueError(msg)
    if not cells:
        return {}
    table: Mapping[Cell, int] = (
        counts_by_cell(train, edges=frame.edges) if counts is None else counts
    )
    fused = fused_shares(train, cells, name=name, rows=rows, frame=frame)
    found: dict[Cell, Ask] = {}
    for cell in cells:
        count = table[cell]
        covered = coverage(fused[cell])
        expected = expected_size(cell, table)
        if rule is QuotaRule.RECOMMENDED:
            asked = recommended(count, covered, expected)
        elif rule is QuotaRule.EXPECTED:
            asked = max(0, expected - count)
        elif rule in MULTIPLES:
            asked = max(0, MULTIPLES[rule] * covered - count)
        else:
            msg = f"no quota rule is named {rule!r}"
            raise ValueError(msg)
        found[cell] = Ask(count, covered, expected, asked)
    return found


def quota(
    train: pd.DataFrame,
    cells: Sequence[Cell],
    rule: QuotaRule,
    *,
    name: str,
    rows: Sequence[FieldMapping],
    frame: FrameSpec,
) -> dict[Cell, int]:
    """The rows each of `cells` is asked for under `rule`, leaving out each cell asked 0.

    Raises:
        ValueError: As `asks`.
        GapError: As `asks`.
    """
    found = asks(train, cells, rule, name=name, rows=rows, frame=frame)
    return {cell: ask.asked for cell, ask in found.items() if ask.asked}
