"""Each numeric field's prompt bounds: the training split's minimum and maximum, age in the band."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bko.baselines.records import Drop, drops
from bko.baselines.request import Request
from bko.datasets.errors import ContractError
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
    with pytest.raises(ContractError, match="holds no record"):
        ranges(TRAIN_A.iloc[:0], UCI_FRAME)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_ranges_refuse_a_non_finite_bound_naming_the_field(bad):
    broken = TRAIN_A.astype({"chol": float}).assign(chol=bad)
    with pytest.raises(ContractError, match="chol has no finite range"):
        ranges(broken, UCI_FRAME)


def test_intervals_are_floats_so_age_is_never_mixed():
    for cell in (YOUNG, OLDER):
        found = intervals(cell, TRAIN_A, frame=UCI_FRAME)
        assert all(isinstance(bound, float) for pair in found.values() for bound in pair), cell


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
    with pytest.raises(ContractError, match="does not overlap"):
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


def _probes(name: str, bound: float, step: float, *, up: bool) -> dict[str, float]:
    return {name: bound + step if up else bound - step}


def test_every_interval_is_what_the_drop_rule_enforces():
    """Each bound keeps its row, and the first value past it drops, for both bands."""
    split_age = (float(TRAIN_A["age"].min()), float(TRAIN_A["age"].max()))
    steps = {"age": 1.0, "trestbps": 1.0, "chol": 1.0, "thalach": 1.0, "oldpeak": 0.1}
    for cell in (YOUNG, OLDER):
        bounds = intervals(cell, TRAIN_A, frame=UCI_FRAME)
        request = Request(quota={cell: 100}, frame=UCI_FRAME)
        probes: list[tuple[dict[str, float], Drop | None]] = []
        for name, (low, high) in bounds.items():
            probes += [({name: low}, None), ({name: high}, None)]
            if name == "age":
                # A band edge drops as another cell; the split's own extreme drops as out of range.
                for bound, extreme, up in ((low, split_age[0], False), (high, split_age[1], True)):
                    drop = Drop.RANGE if bound == extreme else Drop.CELL
                    probes.append((_probes(name, bound, steps[name], up=up), drop))
            else:
                probes += [
                    (_probes(name, low, steps[name], up=False), Drop.RANGE),
                    (_probes(name, high, steps[name], up=True), Drop.RANGE),
                ]
        rows = pd.DataFrame(
            [_row(**{"age": 45 if cell is YOUNG else 60, **probe}) for probe, _ in probes]
        ).astype(dict(UCI_FRAME.dtypes))
        found = drops(rows, TRAIN_A, request)
        assert found == tuple(drop for _, drop in probes), cell


def _user(train: pd.DataFrame) -> tuple[str, str]:
    cells = record_cells(train, (), name="uci", edges=UCI_FRAME.edges)
    guided = guide(
        train,
        cells,
        (),
        definition=Definition.SUBGROUP,
        fragments=(),
        ontologies={},
        rows=(),
        frame=UCI_FRAME,
    )
    return render(guided[YOUNG], 5, Level.CELL, frame=UCI_FRAME)


def test_a_second_split_renders_other_bounds():
    system_a, user_a = _user(TRAIN_A)
    system_b, user_b = _user(TRAIN_B)
    assert system_a == system_b
    assert "chol 180 to 300 mg/dL" in user_a
    assert "chol 180 to 400 mg/dL" in user_b
    assert user_a.replace("300 mg/dL", "400 mg/dL") == user_b


def test_guide_refuses_a_band_outside_the_splits_ages():
    train = TRAIN_A.assign(age=lambda frame: frame["age"].clip(upper=54))
    cells = record_cells(train, (), name="uci", edges=UCI_FRAME.edges)
    with pytest.raises(ContractError, match="does not overlap the split's ages 35 to 54"):
        guide(
            train,
            cells,
            (),
            definition=Definition.SUBGROUP,
            fragments=(),
            ontologies={},
            rows=(),
            frame=UCI_FRAME,
        )


def test_the_age_interval_clips_a_custom_edge_set_to_the_split():
    frame = replace(UCI_FRAME, edges=(40, 60))
    middle = intervals(("40-59", "female", 1), TRAIN_A, frame=frame)
    assert middle["age"] == (40, 59)
    assert intervals(("<40", "female", 1), TRAIN_A, frame=frame)["age"] == (35, 39)
    assert intervals((">=60", "female", 1), TRAIN_A, frame=frame)["age"] == (60, 70)
    with pytest.raises(ValueError, match="not one of"):
        intervals(("<55", "female", 1), TRAIN_A, frame=frame)
