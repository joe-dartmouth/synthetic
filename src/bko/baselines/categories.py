"""The categories `eb-marginals` and `hb-marginals` count a column in, read from its label group.

A coded column's categories are its codes, each code with fewer than `MIN_RECORDS` label-group
records folded into `OTHER`. A measured column's are its values when the label group holds at most
`MAX_VALUES` distinct ones, else the label group's decile bins; walking those up from the lowest,
a category closes once it holds `MIN_RECORDS` label-group records, and an upper tail that never
does joins the last closed one. Every category of a measured column therefore holds `MIN_RECORDS`
label-group records unless the whole group holds fewer; a coded column's `OTHER` can hold fewer,
since it gathers whatever its rare codes hold.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.errors import BaselineError


MIN_RECORDS: Final = 5
"""The label-group records a kept code, or a measured column's category, holds at least, when
the label group holds that many."""

MAX_VALUES: Final = 16
"""A measured column with at most this many distinct label-group values keeps its values."""

DECILES: Final = tuple(step / 10 for step in range(1, 10))
"""The quantiles whose label-group values split a wider measured column into ten bins."""

OTHER: Final = "other"
"""The category a coded column's rare codes fold into."""


def categorise(values: pd.Series, group: pd.Series, *, ordered: bool) -> pd.Series:
    """Each of `values`' category under the categories `group`, the label group, defines.

    Args:
        values: The values to categorise; any index.
        group: The column's label-group values, which fix the categories.
        ordered: True for a measured column, whose categories are value ranges ascending; False
            for a coded one, whose categories are its codes and `OTHER`.

    Returns:
        One category per value on `values`' index: an ordinal from 0 for an ordered column, a code
        or `OTHER` for an unordered one.

    Raises:
        BaselineError: Naming the column, `values`' name, if `values` or `group` holds a missing
            value, which no category holds.
    """
    if values.isna().any() or group.isna().any():
        msg = f"column {values.name} holds a missing value, which no category holds"
        raise BaselineError(msg)
    if not ordered:
        counts = group.value_counts()
        kept = set(counts.index[counts >= MIN_RECORDS])
        return values.map(lambda code: code if code in kept else OTHER).astype(object)
    distinct = np.unique(group.to_numpy(dtype="float64"))
    if len(distinct) <= MAX_VALUES:
        edges = distinct[1:]
    else:
        edges = np.unique(np.quantile(group.to_numpy(dtype="float64"), DECILES))
    atoms = np.bincount(_atom(group, edges), minlength=len(edges) + 1)
    merged = _merge(atoms.tolist())
    return pd.Series(merged[_atom(values, edges)], index=values.index, dtype=object)


def _atom(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Each value's position among the ranges `edges` start, the first open below."""
    return np.searchsorted(edges, values.to_numpy(dtype="float64"), side="right")


def _merge(counts: list[int]) -> np.ndarray:
    """Each atom's category, atoms joined upward until each category holds `MIN_RECORDS`."""
    category = np.zeros(len(counts), dtype="int64")
    current, held, closed = 0, 0, -1
    for position, count in enumerate(counts):
        category[position] = current
        held += count
        if held >= MIN_RECORDS:
            closed, current, held = current, current + 1, 0
    if current > closed >= 0:
        category[category == current] = closed
    return category
