"""`conditional_cart`: visited columns redrawn from same-leaf donors, given columns kept."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from bko.baselines.classical import fit_tree
from bko.baselines.conditional import conditional_cart
from bko.datasets.frames import FrameSpec, RangeConstraint


PAIRED = FrameSpec(
    name="paired",
    features=("age", "sex", "flag", "dose"),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", "flag", "dose"),
    codebook={
        "sex": {0: "female", 1: "male"},
        "flag": {0: "off", 1: "on"},
        "dose": {0: "none", 1: "low", 2: "mid", 3: "high"},
    },
    descriptions={
        "age": "age",
        "sex": "sex",
        "flag": "a flag no given column decides",
        "dose": "twice the flag plus the label",
        "target": "the label",
    },
    ranges=(RangeConstraint("age", "a", 30, 90, source="test", limits="test"),),
    label_codes=(0, 1),
    label_meanings={0: "absent", 1: "present"},
    edges=(45, 70),
    study="a conditional draw",
    cohort="test record",
)
"""A frame whose `dose` is `2 * flag + target` and whose `flag` no given column decides."""

GIVEN = ("age", "sex", "target")
VISIT = ("flag", "dose")


def _train() -> pd.DataFrame:
    """20 records per (sex, target), ids from 101, each age once per `flag`, `dose` fixed.

    Every age, sex and label holds both flags equally, so no split of `flag`'s tree cuts its
    impurity and its one leaf holds every training row; `dose`'s tree splits on `flag`.
    """
    rows = [
        {"age": 30 + (i * 7) % 60, "sex": sex, "flag": flag, "dose": 2 * flag + target}
        | {"target": target}
        for sex in (0, 1)
        for target in (0, 1)
        for i in range(10)
        for flag in (0, 1)
    ]
    frame = pd.DataFrame(rows, columns=list(PAIRED.columns)).astype(dict(PAIRED.dtypes))
    frame.index = pd.RangeIndex(101, 101 + len(frame), name="record_id")
    return frame


def _drawn(size: int = 12) -> pd.DataFrame:
    """`size` rows over every (sex, target), each `flag` 0 and `dose` 3, so a redraw shows."""
    rows = [
        {"age": 40 + k % 40, "sex": k % 2, "flag": 0, "dose": 3, "target": k // 2 % 2}
        for k in range(size)
    ]
    return pd.DataFrame(rows, columns=list(PAIRED.columns)).astype(dict(PAIRED.dtypes))


def _redrawn(drawn: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    return conditional_cart(
        _train(),
        drawn,
        given=GIVEN,
        visit=VISIT,
        frame=PAIRED,
        rng=np.random.default_rng(seed),
        seed=0,
    )


def test_each_value_is_a_training_value_from_the_rows_leaf():
    train, drawn = _train(), _drawn()
    rows, donors = _redrawn(drawn)
    columns = [*GIVEN, *VISIT]
    for at, column in enumerate(VISIT, start=len(GIVEN)):
        predictors = columns[:at]
        _, production = fit_tree(train, columns, at, PAIRED, seed=0)
        oracle = DecisionTreeClassifier(**production.get_params())
        oracle.fit(train[predictors].to_numpy(float), train[column])
        ours = oracle.apply(rows[predictors].to_numpy(float))
        theirs = oracle.apply(train.loc[donors[column], predictors].to_numpy(float))
        assert (ours == theirs).all()
        assert rows[column].tolist() == train.loc[donors[column], column].tolist()
    assert rows["dose"].tolist() == (2 * rows["flag"] + rows["target"]).tolist()
    assert set(donors.columns) == set(VISIT)
    assert donors.isin(train.index).all().all()


def test_a_later_column_reads_the_redrawn_earlier_one():
    drawn = _drawn(40)
    rows, _ = _redrawn(drawn)
    moved = rows["flag"] != drawn["flag"]
    assert moved.any()
    assert rows["dose"].tolist() == (2 * rows["flag"] + rows["target"]).tolist()
    assert (
        rows.loc[moved, "dose"] != 2 * drawn.loc[moved, "flag"] + drawn.loc[moved, "target"]
    ).all()


def test_an_impure_leaf_spreads_its_donors():
    train = _train()
    rows, donors = _redrawn(_drawn(200))
    assert donors["flag"].nunique() >= 2
    assert set(rows["flag"]) == {0, 1}
    assert set(train.loc[donors["flag"], "sex"]) == {0, 1}


def test_the_given_columns_are_untouched():
    drawn = _drawn()
    rows, _ = _redrawn(drawn)
    pd.testing.assert_frame_equal(rows[list(GIVEN)], drawn[list(GIVEN)])
    assert not rows[list(VISIT)].equals(drawn[list(VISIT)])
    assert list(rows.columns) == list(PAIRED.columns)


def test_changing_a_given_value_changes_the_leaf():
    drawn = _drawn()
    flipped = drawn.assign(target=1 - drawn["target"])
    rows, _ = _redrawn(drawn)
    moved, _ = _redrawn(flipped)
    assert moved["flag"].tolist() == rows["flag"].tolist()
    assert rows["dose"].tolist() == (2 * rows["flag"] + drawn["target"]).tolist()
    assert moved["dose"].tolist() == (2 * rows["flag"] + 1 - drawn["target"]).tolist()


def test_the_draw_is_deterministic_per_seed():
    drawn = _drawn()
    first, first_donors = _redrawn(drawn, seed=3)
    again, again_donors = _redrawn(drawn, seed=3)
    _, other_donors = _redrawn(drawn, seed=4)
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first_donors, again_donors)
    assert not first_donors.equals(other_donors)


def test_no_drawn_row_fits_no_tree_and_names_no_donor():
    rows, donors = _redrawn(_drawn().iloc[:0])
    assert rows.empty
    assert list(donors.columns) == list(VISIT)
    assert donors.empty


@pytest.mark.parametrize("size", [0, 12], ids=["empty", "drawn"])
def test_the_donors_are_int64_on_every_path(size):
    _, donors = _redrawn(_drawn(size))
    assert list(donors.columns) == list(VISIT)
    assert all(dtype == np.int64 for dtype in donors.dtypes)


@pytest.mark.parametrize(
    ("given", "visit"),
    [((), VISIT), (GIVEN, ("sex", "dose"))],
    ids=["no-given", "overlap"],
)
def test_a_bad_column_split_is_refused(given, visit):
    with pytest.raises(ValueError, match="must be distinct columns of the frame"):
        conditional_cart(
            _train(),
            _drawn(),
            given=given,
            visit=visit,
            frame=PAIRED,
            rng=np.random.default_rng(0),
            seed=0,
        )


@pytest.mark.parametrize(
    ("given", "visit"),
    [(GIVEN, ("flag", "weight")), (("age", "weight"), VISIT)],
    ids=["visit", "given"],
)
def test_a_column_outside_the_frame_is_refused_by_name(given, visit):
    with pytest.raises(ValueError, match="column 'weight' is not a column of frame 'paired'"):
        conditional_cart(
            _train(),
            _drawn(),
            given=given,
            visit=visit,
            frame=PAIRED,
            rng=np.random.default_rng(0),
            seed=0,
        )
