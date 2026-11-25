"""Coded states: a measured value read as a whole-number state, so a missing value is a state too.

A frame keeps a lab or vital with a cited cutoff, and a numeric field with missing values and no
cited cutoff, as a coded field: `0` = not measured, then one state per band between cutoffs, so
a field row on an abnormal state never matches a missing value. A cutoff read as "at or above"
puts the value at it in the upper state; one read as "above" keeps it in the lower.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd


def coded(
    values: pd.Series, *, at_or_above: Sequence[float] = (), above: Sequence[float] = ()
) -> pd.Series:
    """Each value's state: 0 when missing, else 1 plus the cutoffs it reaches.

    Args:
        values: The measured values; a missing value is NaN.
        at_or_above: Cutoffs a value reaches when it is at or above one, strictly increasing.
        above: Cutoffs a value reaches only when it is strictly above one, strictly increasing.

    Returns:
        The states as int64, on `values`' index: 0 for a missing value, else 1 plus the number of
        `at_or_above` cutoffs at or below the value plus the number of `above` cutoffs strictly
        below it.

    Raises:
        ValueError: If either cutoff sequence is not strictly increasing.
    """
    for name, cutoffs in (("at_or_above", at_or_above), ("above", above)):
        if any(low >= high for low, high in pairwise(cutoffs)):
            msg = f"{name} cutoffs {tuple(cutoffs)} are not strictly increasing"
            raise ValueError(msg)
    measured = values.astype("float64").to_numpy()
    states = np.ones(len(measured), dtype="int64")
    for cutoff in at_or_above:
        states += measured >= cutoff
    for cutoff in above:
        states += measured > cutoff
    states[np.isnan(measured)] = 0
    return pd.Series(states, index=values.index, name=values.name, dtype="int64")


def tertiles(values: pd.Series) -> tuple[float, float]:
    """The one-third and two-thirds quantiles of the measured values, linearly interpolated.

    Raises:
        ValueError: If no value is measured.
    """
    measured = values.dropna().astype("float64")
    if measured.empty:
        msg = f"{values.name or 'the values'}: no measured value to take tertiles of"
        raise ValueError(msg)
    low, high = measured.quantile([1 / 3, 2 / 3]).tolist()
    return float(low), float(high)
