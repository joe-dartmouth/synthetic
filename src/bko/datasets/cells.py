"""The age-sex subgroup-label cells, and which hold too few records.

RESEARCH section 2 sets the rule: a cell is underrepresented when it holds fewer than 20 training
records. A cell is an age band, a sex and a label. Every cell of the grid is reported, those with
no record included, because an empty cell is the most underrepresented of all. The functions read
any frame with integer `age`, `sex` (0 female, 1 male) and `target` (0 or 1) columns, so a Synthea
population shaped that way is counted the same way. ADR `research/uci-evaluation-frame` records why
the bands split at 55.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from typing import Final

import pandas as pd

from bko.datasets.errors import ContractError
from bko.datasets.uci import CODEBOOK, LABEL


UNDERREPRESENTED_BELOW: Final = 20
"""RESEARCH section 2: a cell with fewer training records than this is underrepresented."""

AGE_BAND_EDGES: Final = (55,)
"""Each edge starts a band, so `(55,)` gives the bands `<55` and `>=55`."""

CELL_COLUMNS: Final = ("age_band", "sex", LABEL)
"""The columns that name a cell."""

COUNT_COLUMNS: Final = (*CELL_COLUMNS, "count", "underrepresented")
"""The columns of `cell_counts`."""


def age_band_labels(edges: Sequence[int] = AGE_BAND_EDGES) -> tuple[str, ...]:
    """Return the band labels for `edges`, youngest first: `<a`, then `a-b`, then `>=z`.

    Args:
        edges: Strictly increasing band starts.

    Returns:
        One label per band; a middle band `[a, c)` is labelled `a-(c-1)`.

    Raises:
        ValueError: If `edges` is empty or not strictly increasing.
    """
    if not edges or list(edges) != sorted(set(edges)):
        msg = f"age band edges must be non-empty and strictly increasing, not {tuple(edges)}"
        raise ValueError(msg)
    middle = [f"{low}-{high - 1}" for low, high in itertools.pairwise(edges)]
    return (f"<{edges[0]}", *middle, f">={edges[-1]}")


def cell_keys(frame: pd.DataFrame, *, edges: Sequence[int] = AGE_BAND_EDGES) -> pd.DataFrame:
    """Return each row's cell, on the same index: `age_band` and `sex` as labels, `target` as 0/1.

    Args:
        frame: Records with integer `age`, `sex` and `target`.
        edges: Age band starts.

    Returns:
        A frame with the `CELL_COLUMNS`.

    Raises:
        ContractError: If a `sex` or `target` value is outside its codes.
    """
    sexes = CODEBOOK["sex"]
    for column, codes in (("sex", list(sexes)), (LABEL, [0, 1])):
        outside = ~frame[column].isin(codes)
        if outside.any():
            first = frame.index[outside][0]
            msg = f"{column} of record {first} is {frame.loc[first, column]}, outside {codes}"
            raise ContractError(msg)
    labels = age_band_labels(edges)
    return pd.DataFrame(
        {
            "age_band": [labels[bisect.bisect_right(edges, age)] for age in frame["age"]],
            "sex": frame["sex"].map(dict(sexes)),
            LABEL: frame[LABEL].astype("int64"),
        },
        index=frame.index,
    )


def cell_counts(
    frame: pd.DataFrame,
    *,
    edges: Sequence[int] = AGE_BAND_EDGES,
    below: int = UNDERREPRESENTED_BELOW,
) -> pd.DataFrame:
    """Count the records in every cell of the grid, empty cells included.

    Args:
        frame: Records with integer `age`, `sex` and `target`.
        edges: Age band starts.
        below: A cell with fewer records than this is underrepresented.

    Returns:
        One row per cell in grid order (band, then female before male, then 0 before 1), with
        the `COUNT_COLUMNS`.
    """
    grid = pd.MultiIndex.from_product(
        [list(age_band_labels(edges)), list(CODEBOOK["sex"].values()), [0, 1]],
        names=list(CELL_COLUMNS),
    )
    sizes = cell_keys(frame, edges=edges).groupby(list(CELL_COLUMNS)).size()
    counts = sizes.reindex(grid, fill_value=0).rename("count").reset_index()
    counts["underrepresented"] = counts["count"] < below
    return counts
