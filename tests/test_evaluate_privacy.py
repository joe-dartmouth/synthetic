"""Distance to closest record: copies sit at zero, one differing code moves a row 1/14, and the
blocked computation returns SDMetrics' `calculate_dcr` bit for bit; the second-nearest distance,
spans from another frame, and the nearest-neighbour distance ratio."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import privacy
from bko.evaluate.data import DIABETES_FRAME, SYNTHEA_FRAME
from bko.evaluate.fidelity import metadata
from bko.evaluate.privacy import (
    BLOCK,
    distances,
    distances_and_nndr,
    nearest,
    nearest_from,
    nearest_two,
    nndr,
)
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


def random_records(frame: FrameSpec, count: int, seed: int) -> pd.DataFrame:
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
    train = random_records(frame, 1_100, seed=1)
    rows = pd.concat([random_records(frame, 140, seed=2), train.iloc[:10]], ignore_index=True)
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
    more = random_records(UCI_FRAME, BLOCK + 37, seed=5)
    few = random_records(UCI_FRAME, 23, seed=6)
    rows, train = (more, few) if crossing == "rows" else (few, more)
    expected = _sdmetrics(rows, train, UCI_FRAME)
    assert np.array_equal(nearest(rows, train, frame=UCI_FRAME), expected)


@pytest.mark.parametrize("block", [13, 64, 1_000, None], ids=lambda b: f"block={b}")
def test_the_first_of_the_two_nearest_is_nearest_bitwise(block):
    rows, train = _cases(DIABETES_FRAME)
    size = block or len(train) + len(rows)
    first, _ = nearest_two(rows, train, frame=DIABETES_FRAME, block=size)
    assert np.array_equal(first, nearest(rows, train, frame=DIABETES_FRAME))


def _matrix(rows: pd.DataFrame, train: pd.DataFrame, frame: FrameSpec) -> np.ndarray:
    """Each row's distance to each training row, every column's span the whole of `train`'s."""
    return np.column_stack(
        [
            nearest_from(rows, train.iloc[[j]], frame=frame, spans_from=train)
            for j in range(len(train))
        ]
    )


@pytest.mark.parametrize("block", [7, 13, 64])
def test_the_second_of_the_two_nearest_is_the_second_smallest_distance(block):
    # At 13, the 40 training rows end in a block of one.
    train = random_records(UCI_FRAME, 40, seed=3)
    rows = pd.concat([random_records(UCI_FRAME, 30, seed=4), train.iloc[:5]], ignore_index=True)
    first, second = nearest_two(rows, train, frame=UCI_FRAME, block=block)
    ordered = np.sort(_matrix(rows, train, UCI_FRAME), axis=1)
    assert np.array_equal(first, ordered[:, 0])
    assert np.array_equal(second, ordered[:, 1])


def test_two_nearest_skip_a_missing_distance_and_count_a_missing_second(monkeypatch, caplog):
    planted = (np.array([[np.nan, 0.3, 0.1], [0.2, np.nan, np.nan]]), np.array([[0.05], [np.nan]]))

    def blocks(*_args: object) -> Iterator[tuple[int, int, Iterator[np.ndarray]]]:
        yield 0, 2, iter(planted)

    monkeypatch.setattr(privacy, "_blocks", blocks)
    train = random_records(UCI_FRAME, 4, seed=1)
    first, second = nearest_two(train.iloc[:2], train, frame=UCI_FRAME)
    assert first.tolist() == [0.05, 0.2]
    assert second[0] == pytest.approx(0.1)
    assert np.isnan(second[1])
    nndr(train.iloc[:2], train, frame=UCI_FRAME)
    assert "privacy nndr rows=2 missing_second=1" in caplog.text


def test_two_nearest_need_two_training_rows():
    rows, train = _cases(UCI_FRAME)
    with pytest.raises(ValueError, match="two training rows"):
        nearest_two(rows, train.iloc[:1], frame=UCI_FRAME)


def test_spans_from_the_reference_itself_are_nearest_bitwise():
    rows, train = _cases(DIABETES_FRAME)
    found = nearest_from(rows, train, frame=DIABETES_FRAME, spans_from=train)
    assert np.array_equal(found, nearest(rows, train, frame=DIABETES_FRAME))


def test_a_span_is_read_from_the_training_split_not_the_reference():
    record = FRAME.iloc[[0]].reset_index(drop=True)
    moved = record.assign(age=record["age"] + 10)
    ages = pd.Series(np.linspace(30, 70, len(FRAME)).round(), index=FRAME.index)
    train = FRAME.assign(age=ages.astype(FRAME["age"].dtype))
    span = float(train["age"].max() - train["age"].min())
    found = nearest_from(record, moved, frame=UCI_FRAME, spans_from=train)
    assert found[0] == pytest.approx(10 / span / len(UCI_FRAME.columns))
    # Against the one moved row alone, its own span is 0, so any gap reads 1.
    assert nearest(record, moved, frame=UCI_FRAME)[0] == pytest.approx(1 / len(UCI_FRAME.columns))


def test_nndr_reads_a_tie_at_zero_as_one_and_an_exact_copy_as_zero():
    train = random_records(UCI_FRAME, 50, seed=7)
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    rows = train.iloc[[0, 1]].reset_index(drop=True)
    ratios = pd.Series([1.0, 0.0])
    assert nndr(rows, train, frame=UCI_FRAME) == {
        Measure.NNDR_P5: float(ratios.quantile(0.05)),
        Measure.NNDR_MEDIAN: 0.5,
        Measure.NNDR_TIES: 1.0,
        Measure.NNDR_EXACT: 1.0,
    }


def test_nndr_quantiles_follow_pandas_on_the_ratio_of_the_two_nearest():
    rows, train = _cases(UCI_FRAME)
    first, second = nearest_two(rows, train, frame=UCI_FRAME)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = pd.Series(np.where(second == 0, 1.0, first / second))
    assert nndr(rows, train, frame=UCI_FRAME) == {
        Measure.NNDR_P5: float(ratios.quantile(0.05)),
        Measure.NNDR_MEDIAN: float(ratios.median()),
        Measure.NNDR_TIES: float((second == 0).sum()),
        Measure.NNDR_EXACT: float(((first == 0) & (second > 0)).sum()),
    }
    assert 0 < (first == 0).sum() < len(rows)


def test_nndr_against_one_training_row_is_refused():
    rows, train = _cases(UCI_FRAME)
    with pytest.raises(ValueError, match="two training rows"):
        nndr(rows, train.iloc[:1], frame=UCI_FRAME)


def test_nndr_of_no_rows_is_undefined():
    rows, train = _cases(UCI_FRAME)
    scores = nndr(rows.iloc[:0], train, frame=UCI_FRAME)
    assert all(np.isnan(value) for value in scores.values())
    assert set(scores) == {
        Measure.NNDR_P5,
        Measure.NNDR_MEDIAN,
        Measure.NNDR_TIES,
        Measure.NNDR_EXACT,
    }


def test_one_pass_gives_distances_then_nndr_bitwise():
    rows, train = _cases(DIABETES_FRAME)
    found = distances_and_nndr(rows, train, frame=DIABETES_FRAME)
    expected = {
        **distances(rows, train, frame=DIABETES_FRAME),
        **nndr(rows, train, frame=DIABETES_FRAME),
    }
    assert list(found) == list(expected)
    assert found == expected
