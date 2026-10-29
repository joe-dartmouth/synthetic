"""`bko.fusion.join`: which validated records join the training split, and what it counts."""

from __future__ import annotations

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.fusion.join import join
from bko.gaps.observe import Cell
from bko.validate.checks import validate
from tests._fusion_fixtures import CELL0, CELL1, UCI_X, training_split, young
from tests._validate_fixtures import planted


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

REQUEST: dict[Cell, int] = {CELL0: 20, CELL1: 20}


@pytest.fixture
def train(tmp_path):
    return training_split(tmp_path, UCI_X)


def test_accepted_records_follow_every_training_record(train):
    joined = join(
        train,
        {CELL1: validate([young(1, cp=1)] * 20, frame=UCI_FRAME).accepted},
        REQUEST,
        edges=UCI_FRAME.edges,
    )
    frame = joined.frame
    assert len(frame) == len(train) + 20
    assert frame.iloc[: len(train)].equals(train)
    assert list(frame.index[len(train) :]) == list(range(-1, -21, -1))
    assert frame.index.name == train.index.name
    assert frame.dtypes.equals(train.dtypes)
    assert dict(joined.joined) == {CELL0: 0, CELL1: 20}


def test_an_off_cell_record_is_dropped_and_counted(train):
    items = [young(1, cp=1)] * 19 + [planted(age=60, sex=0, target=1, cp=1)]
    joined = join(
        train, {CELL1: validate(items, frame=UCI_FRAME).accepted}, REQUEST, edges=UCI_FRAME.edges
    )
    assert joined.off_cell[CELL1] == 1
    assert joined.joined[CELL1] == 19


def test_a_record_past_its_cells_rows_is_dropped_and_counted(train):
    items = [young(1, cp=1, chol=200 + i) for i in range(22)]
    joined = join(
        train, {CELL1: validate(items, frame=UCI_FRAME).accepted}, REQUEST, edges=UCI_FRAME.edges
    )
    assert joined.past_quota[CELL1] == 2
    assert list(joined.frame["chol"].iloc[len(train) :]) == list(range(200, 220))


def test_a_cell_with_no_accepted_records_joins_nothing(train):
    joined = join(train, {}, REQUEST, edges=UCI_FRAME.edges)
    assert joined.frame.equals(train)
    assert dict(joined.joined) == {CELL0: 0, CELL1: 0}
    assert dict(joined.off_cell) == {CELL0: 0, CELL1: 0}
    assert dict(joined.past_quota) == {CELL0: 0, CELL1: 0}


def test_records_for_a_cell_the_request_does_not_ask_for_are_refused(train):
    accepted = {CELL1: validate([young(1, cp=1)], frame=UCI_FRAME).accepted}
    with pytest.raises(ValueError, match="does not ask for: <55/female/1"):
        join(train, accepted, {CELL0: 20}, edges=UCI_FRAME.edges)


def test_a_frame_without_the_training_splits_columns_is_refused(train):
    accepted = {CELL1: validate([young(1, cp=1)], frame=UCI_FRAME).accepted.drop(columns="chol")}
    with pytest.raises(ValueError, match="differ from the training split's columns or dtypes"):
        join(train, accepted, REQUEST, edges=UCI_FRAME.edges)


def test_the_training_frame_is_left_unchanged(train):
    before = train.copy()
    join(
        train,
        {CELL1: validate([young(1, cp=1)] * 20, frame=UCI_FRAME).accepted},
        REQUEST,
        edges=UCI_FRAME.edges,
    )
    assert train.equals(before)
    assert train.index.equals(before.index)


@pytest.mark.parametrize(
    "stray",
    [
        planted(age=60, sex=0, target=1, cp=1),
        planted(age=40, sex=1, target=1, cp=1),
        planted(age=40, sex=0, target=0, cp=1),
    ],
    ids=["band", "sex", "target"],
)
def test_a_record_differing_in_any_one_cell_part_is_off_cell(train, stray):
    items = [young(1, cp=1)] * 19 + [stray]
    joined = join(
        train, {CELL1: validate(items, frame=UCI_FRAME).accepted}, REQUEST, edges=UCI_FRAME.edges
    )
    assert joined.off_cell[CELL1] == 1
    assert joined.joined[CELL1] == 19


def test_the_age_band_edge_places_a_record(train):
    items = [planted(age=54, sex=0, target=1, cp=1), planted(age=55, sex=0, target=1, cp=1)]
    joined = join(
        train, {CELL1: validate(items, frame=UCI_FRAME).accepted}, REQUEST, edges=UCI_FRAME.edges
    )
    assert (joined.joined[CELL1], joined.off_cell[CELL1]) == (1, 1)
    wide: Cell = ("<56", "female", 1)
    moved = join(train, {wide: validate(items, frame=UCI_FRAME).accepted}, {wide: 20}, edges=(56,))
    assert (moved.joined[wide], moved.off_cell[wide]) == (2, 0)


def test_joined_records_follow_request_order_whatever_the_accepted_order(train):
    accepted = {
        CELL1: validate([young(1, cp=1, chol=300)] * 2, frame=UCI_FRAME).accepted,
        CELL0: validate([young(0, cp=4, chol=200)] * 3, frame=UCI_FRAME).accepted,
    }
    frame = join(train, accepted, REQUEST, edges=UCI_FRAME.edges).frame
    added = frame.iloc[len(train) :]
    assert list(added["chol"]) == [200, 200, 200, 300, 300]
    assert list(added.index) == [-1, -2, -3, -4, -5]


def test_a_frame_with_another_dtype_is_refused(train):
    frame = validate([young(1, cp=1)], frame=UCI_FRAME).accepted
    frame["chol"] = frame["chol"].astype("float64")
    with pytest.raises(ValueError, match="columns or dtypes"):
        join(train, {CELL1: frame}, REQUEST, edges=UCI_FRAME.edges)


def test_join_requires_edges():
    with pytest.raises(TypeError, match="edges"):
        join(pd.DataFrame(), {}, {})  # ty: ignore[missing-argument]
