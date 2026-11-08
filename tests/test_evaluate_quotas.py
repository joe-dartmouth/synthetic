"""`bko.evaluate.quotas`: a rewrite call's per-field value counts from its label group's rows."""

from __future__ import annotations

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.errors import EvaluationError
from bko.evaluate.quotas import Category, allot, apportion, categories, category_of, spread


SIZES = [20] * 62 + [10]
"""The calls of a 1250-row holdout cell at 20 rows a call."""


def _ranges(found: tuple[Category, ...]) -> list[tuple[int | float, int | float, int]]:
    return [(category.low, category.high, category.rows) for category in found]


def test_a_coded_field_takes_every_code_as_a_category():
    reference = pd.DataFrame({"thal": [7, 3, 7, 7]})
    found = categories(reference, ("thal",), UCI_FRAME, {})
    assert _ranges(found["thal"]) == [(3, 3, 1), (6, 6, 0), (7, 7, 3)]


def test_a_measured_range_closes_at_one_percent_and_tiles_the_training_range():
    reference = pd.DataFrame({"chol": [0] * 150 + [1] * 47 + [2] * 2 + [3]})
    found = categories(reference, ("chol",), UCI_FRAME, {"chol": (0.0, 5.0)})
    assert _ranges(found["chol"]) == [(0, 0, 150), (1, 1, 47), (2, 5, 3)]


def test_measured_ranges_cover_values_between_and_beyond_the_label_groups():
    reference = pd.DataFrame({"chol": [0] * 50 + [4] * 50 + [9] * 50})
    found = categories(reference, ("chol",), UCI_FRAME, {"chol": (0.0, 12.0)})
    assert [(c.low, c.high) for c in found["chol"]] == [(0, 0), (1, 4), (5, 12)]


def test_apportion_sums_to_the_rows_and_breaks_ties_to_the_earlier_category():
    found = (Category(0, 0, 1), Category(1, 1, 1), Category(2, 2, 1))
    assert apportion(found, 4) == (2, 1, 1)


def test_spread_gives_each_call_its_size_and_the_holdout_its_apportionment():
    counts = (11, 0, 25, 13)
    sizes = (20, 20, 9)
    calls = spread(counts, sizes)
    assert [sum(call) for call in calls] == list(sizes)
    assert tuple(sum(call[k] for call in calls) for k in range(len(counts))) == counts


def test_spread_refuses_sizes_that_are_not_the_quotas_rows():
    with pytest.raises(ValueError, match="not the quotas'"):
        spread((3, 4), (5,))


def test_a_rare_first_category_lands_in_evenly_spaced_calls():
    calls = spread((3, 1247), SIZES)
    held = [index for index, call in enumerate(calls) for _ in range(call[0])]
    assert len(held) == 3
    for index, expected in zip(held, (10, 31, 52), strict=True):
        assert abs(index - expected) <= 1, held


def test_single_token_categories_land_in_distinct_calls():
    calls = spread((3, 1, 1, 1, 1, 1, 1242), SIZES)
    landed = [
        index for k in range(1, 6) for index, call in enumerate(calls) for _ in range(call[k])
    ]
    assert len(landed) == 5
    assert len(set(landed)) == 5, landed


def test_allot_is_equal_from_shuffled_training_rows():
    reference = pd.DataFrame(
        {
            "chol": [180 + (index * 7) % 120 for index in range(300)],
            "fbs": [index % 3 == 0 for index in range(300)],
            "oldpeak": [round((index * 0.3) % 4.0, 1) for index in range(300)],
        }
    ).astype({"fbs": int})
    fields = ("chol", "fbs", "oldpeak")
    bounds = {"chol": (150.0, 320.0), "oldpeak": (0.0, 5.0)}
    sizes = (20, 20, 9)
    shuffled = reference.sample(frac=1, random_state=1)
    ours = allot(categories(reference, fields, UCI_FRAME, bounds), sizes)
    theirs = allot(categories(shuffled, fields, UCI_FRAME, bounds), sizes)
    assert ours == theirs
    assert [sum(call["fbs"]) for call in ours] == list(sizes)


def test_an_empty_label_group_is_refused():
    with pytest.raises(EvaluationError, match="no training row"):
        categories(pd.DataFrame({"chol": []}), ("chol",), UCI_FRAME, {"chol": (0.0, 5.0)})


def test_a_field_whose_categories_hold_no_row_is_refused_by_name():
    reference = pd.DataFrame({"thal": [9, 9], "chol": [float("nan")] * 2})
    with pytest.raises(EvaluationError, match="'thal'"):
        categories(reference, ("thal",), UCI_FRAME, {})
    with pytest.raises(EvaluationError, match="'chol'"):
        categories(reference, ("chol",), UCI_FRAME, {"chol": (0.0, 5.0)})


def test_category_of_assigns_a_value_to_the_first_range_reaching_it():
    found = (Category(0, 0, 150), Category(1, 1, 47), Category(2, 5, 3))
    assert [category_of(value, found) for value in (0, 1, 2, 5, 9)] == [0, 1, 2, 2, 2]
    fractional = (Category(0.0, 0.5, 10), Category(0.8, 4.0, 10))
    assert category_of(0.6, fractional) == 1
