"""Distance to closest record: copies sit at zero, and one differing code moves a row 1/14."""

from __future__ import annotations

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.privacy import distances
from bko.evaluate.scores import Measure
from tests._baselines_fixtures import full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FRAME = full_records(8)


def test_copies_of_training_rows_sit_at_zero():
    assert distances(FRAME.iloc[:10], FRAME, frame=UCI_FRAME) == {
        Measure.DCR_P5: 0.0,
        Measure.DCR_MEDIAN: 0.0,
    }


def test_one_differing_code_moves_a_row_by_one_column_share():
    train = FRAME.iloc[[0]]
    other = 4 if int(train["cp"].iloc[0]) != 4 else 1
    scores = distances(train.assign(cp=other), train, frame=UCI_FRAME)
    assert scores[Measure.DCR_MEDIAN] == pytest.approx(1 / 14)
    assert scores[Measure.DCR_P5] == pytest.approx(1 / 14)
