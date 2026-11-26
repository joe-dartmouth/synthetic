"""Coded states: a measured value read as a whole-number state, so a missing value is a state too.

A frame keeps a lab or vital with a cited cutoff, and a numeric field with missing values and no
cited cutoff, as a coded field: `0` = not measured, then one state per band between cutoffs, so
a field row on an abnormal state never matches a missing value. A cutoff read as "at or above"
puts the value at it in the upper state; one read as "above" keeps it in the lower. `Cutoffs`
holds one field's cutoffs and their source, `band_meanings` names each state's band, and
`band_state` finds a state by its band's name, so a field row names the band it means.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Cutoffs:
    """Where a measured field's states split (`coded`).

    Attributes:
        at_or_above: Cutoffs a value at or above reaches.
        above: Cutoffs only a value strictly above reaches.
        source: The `docs/literature.md` key the cutoffs are read in; None for cutoffs taken from
            the data themselves, such as tertiles, which no field row may cite.
    """

    at_or_above: tuple[float, ...] = ()
    above: tuple[float, ...] = ()
    source: str | None = None


def band_meanings(cut: Cutoffs) -> Mapping[int, str]:
    """Each state of a field coded at `cut` and what it means, state 0 `not measured`."""
    bounds = sorted([(c, True) for c in cut.at_or_above] + [(c, False) for c in cut.above])
    meanings = {0: "not measured"}
    for state in range(len(bounds) + 1):
        low = bounds[state - 1] if state else None
        high = bounds[state] if state < len(bounds) else None
        parts = []
        if low is not None:
            parts.append(f"at least {low[0]:g}" if low[1] else f"above {low[0]:g}")
        if high is not None:
            parts.append(f"below {high[0]:g}" if high[1] else f"at most {high[0]:g}")
        meanings[state + 1] = " and ".join(parts) or "measured"
    return MappingProxyType(meanings)


def band_state(cut: Cutoffs, band: str) -> int:
    """The state whose band `band_meanings` names `band`, such as `above 90`.

    Raises:
        ValueError: If no state of `cut` has that band, naming the bands it has.
    """
    for state, meaning in band_meanings(cut).items():
        if meaning == band:
            return state
    msg = f"no band {band!r} among {list(band_meanings(cut).values())}"
    raise ValueError(msg)


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
