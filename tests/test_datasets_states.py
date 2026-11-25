"""Coded states: a measured value as 0 when missing, else its band among cited cutoffs."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from bko.datasets.states import coded, tertiles


def _states(values: list[float], **cutoffs: tuple[float, ...]) -> list[int]:
    found = coded(pd.Series(values, dtype="float64"), **cutoffs)
    assert found.dtype == "int64"
    return found.tolist()


def test_a_missing_value_is_state_0() -> None:
    assert _states([math.nan, 1.0], at_or_above=(1.2,)) == [0, 1]


def test_a_value_at_an_at_or_above_cutoff_takes_the_upper_state() -> None:
    # Creatinine "1.2 and above": 1.2 is in the upper state.
    assert _states([1.1, 1.2, 1.3], at_or_above=(1.2,)) == [1, 2, 2]


def test_a_value_at_an_above_cutoff_stays_in_the_lower_state() -> None:
    # Heart rate "above 90": 90 stays in the lower state, 91 is above.
    assert _states([90.0, 91.0], above=(90,)) == [1, 2]


def test_two_sided_cutoffs_give_three_states() -> None:
    # Temperature: below 36, 36 to 38, above 38.
    assert _states([35.9, 36.0, 38.0, 38.1], at_or_above=(36,), above=(38,)) == [1, 2, 2, 3]


def test_no_cutoff_leaves_every_measured_value_in_state_1() -> None:
    assert _states([0.0, 500.0, math.nan]) == [1, 1, 0]


def test_the_index_is_kept() -> None:
    found = coded(pd.Series([1.0, math.nan], index=[7, 3]), above=(0.5,))
    assert found.index.tolist() == [7, 3]
    assert found.tolist() == [2, 0]


@pytest.mark.parametrize(
    "cutoffs",
    [{"at_or_above": (2.0, 1.0)}, {"above": (38, 36)}, {"at_or_above": (1.0, 1.0)}],
)
def test_unsorted_cutoffs_are_refused(cutoffs: dict[str, tuple[float, ...]]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        coded(pd.Series([1.0]), **cutoffs)


def test_tertiles_ignore_missing_values() -> None:
    values = pd.Series([math.nan, 1.0, 2.0, 3.0, 4.0, math.nan, 5.0, 6.0, 7.0])
    # Linear interpolation over the seven measured values: positions 2 and 4 of 0..6.
    assert tertiles(values) == pytest.approx((3.0, 5.0))


def test_tertiles_of_no_measured_value_are_refused() -> None:
    with pytest.raises(ValueError, match="no measured value"):
        tertiles(pd.Series([math.nan, math.nan]))
