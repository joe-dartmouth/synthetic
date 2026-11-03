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
from bko.evaluate.privacy import BLOCK, distances, nearest
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


def test_rows_against_no_training_rows_are_refused():
    rows, train = _cases(UCI_FRAME)
    with pytest.raises(ValueError, match="no training rows to measure against"):
        nearest(rows, train.iloc[:0], frame=UCI_FRAME)


def test_no_rows_have_no_distances():
    rows, train = _cases(UCI_FRAME)
    assert nearest(rows.iloc[:0], train, frame=UCI_FRAME).shape == (0,)


def _missing(frame: FrameSpec, *, numeric: bool, rows: bool, train: bool) -> tuple:
    """`_cases`' rows and training rows with some values of one column missing on each side
    named: the first numeric column, or the first coded one."""
    cases = _cases(frame)
    name = frame.numeric[0] if numeric else next(c for c in frame.columns if c not in frame.numeric)
    found = []
    for frame_rows, blank in zip(cases, (rows, train), strict=True):
        frame_rows = frame_rows.astype({name: "float64"})
        if blank:
            frame_rows.loc[frame_rows.index[::3], name] = np.nan
        found.append(frame_rows)
    return tuple(found)


@pytest.mark.parametrize("numeric", [True, False], ids=["numeric", "coded"])
@pytest.mark.parametrize(
    ("rows", "train"), [(True, False), (False, True), (True, True)], ids=["rows", "train", "both"]
)
def test_blocked_dcr_equals_sdmetrics_with_missing_values(numeric, rows, train):
    data, reference = _missing(UCI_FRAME, numeric=numeric, rows=rows, train=train)
    expected = _sdmetrics(data, reference, UCI_FRAME)
    assert np.array_equal(nearest(data, reference, frame=UCI_FRAME), expected)


def test_a_wholly_missing_nullable_training_column_measures_as_sdmetrics_does():
    rows, train = _cases(UCI_FRAME)
    name = UCI_FRAME.numeric[0]
    train[name] = pd.array([pd.NA] * len(train), dtype="Int64")
    found = nearest(rows, train, frame=UCI_FRAME)
    assert np.array_equal(found, _sdmetrics(rows, train.astype({name: "float64"}), UCI_FRAME))


@pytest.mark.parametrize("crossing", ["rows", "train"])
def test_blocked_dcr_equals_sdmetrics_across_the_default_block(crossing):
    """`BLOCK` rows or training rows and a few more, so the default blocks split one side."""
    more = _records(UCI_FRAME, BLOCK + 37, seed=5)
    few = _records(UCI_FRAME, 23, seed=6)
    rows, train = (more, few) if crossing == "rows" else (few, more)
    expected = _sdmetrics(rows, train, UCI_FRAME)
    assert np.array_equal(nearest(rows, train, frame=UCI_FRAME), expected)
