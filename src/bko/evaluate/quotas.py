"""Per-call value quotas for the rewrite arm, from the cell's label-group training rows.

`categories` splits each field's values: every code of a coded field, and for a measured field
contiguous ranges in ascending order, each closed once it holds at least 1% of the rows, the
remainder joining the last. Coded and integer fields' ranges tile the training range; a fractional
field's range starts at the next observed value, and `category_of` assigns a value between ranges
to the next one. `apportion` gives a
holdout's rows to the categories by largest remainder on integers, ties to the earlier category.
`spread` places category k's T tokens at ((j + 1/2) / T + k / K) mod 1, sorts them and cuts the
sequence into the calls' sizes, so a category's calls are evenly spaced and categories with one
token land apart. ADR `research/identifier-perturbation` records the rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError


@dataclass(frozen=True, slots=True)
class Category:
    """A field's values from `low` to `high` inclusive and the label-group rows there."""

    low: int | float
    high: int | float
    rows: int


def categories(
    reference: pd.DataFrame,
    fields: Sequence[str],
    frame: FrameSpec,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[Category, ...]]:
    """Each field's categories over `reference`, the cell's label-group training rows.

    Raises:
        EvaluationError: If `reference` holds no row, or a field's categories hold none.
    """
    total = len(reference)
    if not total:
        msg = "the cell's label group holds no training row, so no quota can be set"
        raise EvaluationError(msg)
    found: dict[str, tuple[Category, ...]] = {}
    for name in fields:
        counts = reference[name].value_counts().sort_index()
        if name in frame.categorical:
            found[name] = tuple(
                Category(code, code, int(counts.get(code, 0)))
                for code in sorted(frame.codebook[name])
            )
            if not sum(category.rows for category in found[name]):
                msg = f"no row of the label group holds a code of {name!r}, so no quota can be set"
                raise EvaluationError(msg)
            continue
        cast = float if name in frame.fractional else int
        observed = [cast(value) for value in counts.index]
        closed: list[tuple[int | float, int]] = []
        held = 0
        for value, count in zip(observed, counts.tolist(), strict=True):
            held += int(count)
            if held * 100 >= total:
                closed.append((value, held))
                held = 0
        if held:
            top, rows = closed[-1]
            closed[-1] = (top, rows + held)
        low, high = bounds[name]
        start: int | float = cast(low)
        ranges: list[Category] = []
        for index, (top, rows) in enumerate(closed):
            last = index == len(closed) - 1
            ranges.append(Category(start, cast(high) if last else top, rows))
            if not last:
                start = top + 1 if cast is int else min(v for v in observed if v > top)
        found[name] = tuple(ranges)
        if not sum(category.rows for category in found[name]):
            msg = f"no row of the label group holds a value of {name!r}, so no quota can be set"
            raise EvaluationError(msg)
    return found


def category_of(value: int | float, found: Sequence[Category]) -> int:
    """The index of the first category whose `high` reaches `value`, else the last."""
    for index, category in enumerate(found):
        if value <= category.high:
            return index
    return len(found) - 1


def apportion(found: Sequence[Category], rows: int) -> tuple[int, ...]:
    """`rows` given to `found` by largest remainder, ties to the earlier category."""
    total = sum(category.rows for category in found)
    base = [rows * category.rows // total for category in found]
    order = sorted(range(len(found)), key=lambda i: (-(rows * found[i].rows % total), i))
    for index in order[: rows - sum(base)]:
        base[index] += 1
    return tuple(base)


def spread(counts: Sequence[int], sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """Each call's count per category, from evenly spaced, offset tokens cut by `sizes`.

    Raises:
        ValueError: If `sizes` and `counts` do not sum to the same rows.
    """
    if sum(sizes) != sum(counts):
        msg = f"the calls' {sum(sizes)} rows are not the quotas' {sum(counts)}"
        raise ValueError(msg)
    width = len(counts)
    tokens = sorted(
        ((Fraction(2 * j + 1, 2 * total) + Fraction(index, width)) % 1, index)
        for index, total in enumerate(counts)
        for j in range(total)
    )
    calls: list[tuple[int, ...]] = []
    start = 0
    for size in sizes:
        held = [0] * width
        for _, index in tokens[start : start + size]:
            held[index] += 1
        calls.append(tuple(held))
        start += size
    return calls


def allot(
    found: Mapping[str, Sequence[Category]], sizes: Sequence[int]
) -> list[dict[str, tuple[int, ...]]]:
    """Each call's count per category of each field, for one cell's calls of `sizes`."""
    rows = sum(sizes)
    per_field = {name: spread(apportion(cats, rows), sizes) for name, cats in found.items()}
    return [{name: calls[i] for name, calls in per_field.items()} for i in range(len(sizes))]
