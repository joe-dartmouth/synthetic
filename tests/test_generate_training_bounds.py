"""Each numeric field's prompt bounds: the training split's minimum and maximum, age in the band."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bko.baselines.records import Drop, drops
from bko.baselines.request import Request
from bko.datasets.split import ranges
from bko.datasets.uci import UCI_FRAME
from bko.gaps.definitions import Definition
from bko.gaps.observe import record_cells
from bko.generate.guidance import guide, intervals
from bko.generate.messages import Level
from bko.generate.prompt import render
from tests._generate_fixtures import TRAIN_A, TRAIN_B
from tests._validate_fixtures import BASE


YOUNG = ("<55", "female", 1)
"""A young cell, whose band caps age below the split's maximum."""

OLDER = (">=55", "female", 1)
"""An older cell, whose band raises age above the split's minimum."""


def test_ranges_are_each_numeric_fields_training_min_and_max():
    assert ranges(TRAIN_A, UCI_FRAME) == {
        "age": (35.0, 70.0),
        "trestbps": (110.0, 155.0),
        "chol": (180.0, 300.0),
        "thalach": (120.0, 174.0),
        "oldpeak": (0.0, 2.7),
    }
    assert ranges(TRAIN_B, UCI_FRAME)["chol"] == (180.0, 400.0)


def test_ranges_refuse_an_empty_split():
    with pytest.raises(ValueError, match="holds no record"):
        ranges(TRAIN_A.iloc[:0], UCI_FRAME)


def test_intervals_follow_the_training_split():
    a = intervals(YOUNG, TRAIN_A, frame=UCI_FRAME)
    b = intervals(YOUNG, TRAIN_B, frame=UCI_FRAME)
    assert (a["chol"], b["chol"]) == ((180.0, 300.0), (180.0, 400.0))
    assert {name: bounds for name, bounds in a.items() if name not in {"age", "chol"}} == {
        name: bounds for name, bounds in b.items() if name not in {"age", "chol"}
    }
    assert list(a) == list(UCI_FRAME.numeric)


def test_the_age_interval_intersects_the_band_and_the_split():
    assert intervals(YOUNG, TRAIN_A, frame=UCI_FRAME)["age"] == (35, 54)
    assert intervals(OLDER, TRAIN_A, frame=UCI_FRAME)["age"] == (55, 70)


def test_a_band_outside_the_splits_ages_is_refused():
    frame = replace(UCI_FRAME, edges=(80,))
    with pytest.raises(ValueError, match="does not overlap"):
        intervals((">=80", "female", 1), TRAIN_A, frame=frame)
    with pytest.raises(ValueError, match="not one of"):
        intervals(("60-69", "female", 1), TRAIN_A, frame=UCI_FRAME)


def _row(**changes: float) -> dict[str, int | float]:
    """A young woman with the condition, every numeric field inside `TRAIN_A`'s range."""
    return {
        **BASE,
        "age": 45,
        "sex": 0,
        "target": 1,
        "trestbps": 130,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
        **changes,
    }


def test_every_interval_is_what_the_drop_rule_enforces():
    bounds = intervals(YOUNG, TRAIN_A, frame=UCI_FRAME)
    request = Request(quota={YOUNG: 100})
    inside = [_row(**{name: value}) for name, pair in bounds.items() for value in pair]
    outside = [_row(**{name: low - 1}) for name, (low, _) in bounds.items() if name != "age"] + [
        _row(**{name: high + 1}) for name, (_, high) in bounds.items() if name != "age"
    ]
    rows = pd.DataFrame([*inside, *outside, _row(age=55)]).astype(dict(UCI_FRAME.dtypes))
    found = drops(rows, TRAIN_A, request)
    assert found[: len(inside)] == (None,) * len(inside)
    assert found[len(inside) : -1] == (Drop.RANGE,) * len(outside)
    assert found[-1] is Drop.CELL


def _user(train: pd.DataFrame) -> tuple[str, str]:
    cells = record_cells(train, (), name="uci")
    guided = guide(
        train, cells, (), definition=Definition.SUBGROUP, fragments=(), ontologies={}, rows=()
    )
    return render(guided[YOUNG], 5, Level.CELL)


def test_a_second_split_renders_other_bounds():
    system_a, user_a = _user(TRAIN_A)
    system_b, user_b = _user(TRAIN_B)
    assert system_a == system_b
    assert "chol 180 to 300 mg/dL" in user_a
    assert "chol 180 to 400 mg/dL" in user_b
    assert user_a.replace("300 mg/dL", "400 mg/dL") == user_b
