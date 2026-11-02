"""Distance to closest record: copies sit at zero, one differing code moves a row 1/14, and the
blocked computation returns SDMetrics' `calculate_dcr` bit for bit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import DIABETES_FRAME, SYNTHEA_FRAME
from bko.evaluate.fidelity import metadata
from bko.evaluate.privacy import distances, nearest
from bko.evaluate.scores import Measure
from tests._baselines_fixtures import full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FRAME = full_records(8)

FRAMES = (UCI_FRAME, SYNTHEA_FRAME, DIABETES_FRAME)


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


def _records(frame: FrameSpec, count: int, seed: int) -> pd.DataFrame:
    """`count` random records in `frame`'s columns: numbers in range, codes from the codebook."""
    rng = np.random.default_rng(seed)
    bounds = {constraint.field: constraint for constraint in frame.ranges}
    columns: dict[str, np.ndarray] = {}
    for name in frame.columns:
        if name in frame.numeric:
            low, high = bounds[name].low, bounds[name].high
            if name in frame.fractional:
                values = rng.uniform(low, high, count).round(frame.fraction_digits)
            else:
                values = rng.integers(int(low), int(high), count, endpoint=True)
            columns[name] = values
        elif name == frame.label:
            columns[name] = rng.choice(np.array(frame.label_codes), count)
        else:
            columns[name] = rng.choice(np.array(list(frame.codebook[name])), count)
    return pd.DataFrame(columns)


def _cases(frame: FrameSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows against training rows that span SDMetrics' chunk, some of them copies of training rows,
    with one numeric column constant in the training rows (a zero range) but not in the rows."""
    train = _records(frame, 1_100, seed=1)
    rows = pd.concat([_records(frame, 140, seed=2), train.iloc[:10]], ignore_index=True)
    constant = frame.numeric[-1]
    train[constant] = train[constant].iloc[0]
    return rows, train


def _sdmetrics(rows: pd.DataFrame, train: pd.DataFrame, frame: FrameSpec) -> np.ndarray:
    columns = list(frame.columns)
    return calculate_dcr(
        dataset=rows.loc[:, columns],
        reference_dataset=train.loc[:, columns],
        metadata=metadata(frame),
    ).to_numpy()


@pytest.mark.parametrize("frame", FRAMES, ids=lambda frame: frame.name)
def test_blocked_dcr_equals_sdmetrics_bitwise(frame: FrameSpec):
    rows, train = _cases(frame)
    expected = _sdmetrics(rows, train, frame)
    assert np.array_equal(nearest(rows, train, frame=frame), expected)


@pytest.mark.parametrize("frame", FRAMES, ids=lambda frame: frame.name)
def test_blocked_dcr_equals_sdmetrics_on_a_zero_range_column_in_both(frame: FrameSpec):
    rows, train = _cases(frame)
    constant = frame.numeric[-1]
    rows[constant] = train[constant].iloc[0]
    rows.loc[rows.index[:5], constant] = train[constant].iloc[0] + 1
    assert np.array_equal(nearest(rows, train, frame=frame), _sdmetrics(rows, train, frame))


def test_blocked_dcr_is_independent_of_block_size():
    rows, train = _cases(DIABETES_FRAME)
    whole = nearest(rows, train, frame=DIABETES_FRAME, block=len(train) + len(rows))
    for block in (13, 64, 1_000):
        assert np.array_equal(nearest(rows, train, frame=DIABETES_FRAME, block=block), whole)
