"""`bko.fusion.join`: which validated records join the training split, and what it counts."""

from __future__ import annotations

import pytest

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
    joined = join(train, {CELL1: validate([young(1, cp=1)] * 20).accepted}, REQUEST)
    frame = joined.frame
    assert len(frame) == len(train) + 20
    assert frame.iloc[: len(train)].equals(train)
    assert list(frame.index[len(train) :]) == list(range(-1, -21, -1))
    assert frame.index.name == train.index.name
    assert frame.dtypes.equals(train.dtypes)
    assert dict(joined.joined) == {CELL0: 0, CELL1: 20}


def test_an_off_cell_record_is_dropped_and_counted(train):
    items = [young(1, cp=1)] * 19 + [planted(age=60, sex=0, target=1, cp=1)]
    joined = join(train, {CELL1: validate(items).accepted}, REQUEST)
    assert joined.off_cell[CELL1] == 1
    assert joined.joined[CELL1] == 19


def test_a_record_past_its_cells_rows_is_dropped_and_counted(train):
    items = [young(1, cp=1, chol=200 + i) for i in range(22)]
    joined = join(train, {CELL1: validate(items).accepted}, REQUEST)
    assert joined.past_quota[CELL1] == 2
    assert list(joined.frame["chol"].iloc[len(train) :]) == list(range(200, 220))


def test_a_cell_with_no_accepted_records_joins_nothing(train):
    joined = join(train, {}, REQUEST)
    assert joined.frame.equals(train)
    assert dict(joined.joined) == {CELL0: 0, CELL1: 0}
    assert dict(joined.off_cell) == {CELL0: 0, CELL1: 0}
    assert dict(joined.past_quota) == {CELL0: 0, CELL1: 0}


def test_records_for_a_cell_the_request_does_not_ask_for_are_refused(train):
    accepted = {CELL1: validate([young(1, cp=1)]).accepted}
    with pytest.raises(ValueError, match="does not ask for: <55/female/1"):
        join(train, accepted, {CELL0: 20})


def test_a_frame_without_the_training_splits_columns_is_refused(train):
    accepted = {CELL1: validate([young(1, cp=1)]).accepted.drop(columns="chol")}
    with pytest.raises(ValueError, match="lack the training split's columns"):
        join(train, accepted, REQUEST)


def test_the_training_frame_is_left_unchanged(train):
    before = train.copy()
    join(train, {CELL1: validate([young(1, cp=1)] * 20).accepted}, REQUEST)
    assert train.equals(before)
    assert train.index.equals(before.index)
