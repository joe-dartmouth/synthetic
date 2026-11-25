"""`conditional_cart`: visited columns redrawn from same-leaf donors, given columns kept."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

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
        "flag": "a flag equal to sex",
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
"""A frame whose `flag` is `sex` and whose `dose` is `2 * flag + target`, so every leaf is pure."""

GIVEN = ("age", "sex", "target")
VISIT = ("flag", "dose")


def _train() -> pd.DataFrame:
    """20 records per (sex, target), ids from 101, with `flag` and `dose` fixed by them."""
    rows = [
        {"age": 30 + (i * 7) % 60, "sex": sex, "flag": sex, "dose": 2 * sex + target}
        | {"target": target}
        for sex in (0, 1)
        for target in (0, 1)
        for i in range(20)
    ]
    frame = pd.DataFrame(rows, columns=list(PAIRED.columns)).astype(dict(PAIRED.dtypes))
    frame.index = pd.RangeIndex(101, 101 + len(frame), name="record_id")
    return frame


def _drawn() -> pd.DataFrame:
    """12 rows over every (sex, target), each `flag` and `dose` wrong, so a redraw shows."""
    rows = [
        {"age": 40 + k, "sex": k % 2, "flag": 1 - k % 2, "dose": 3 - (k % 2) * 3, "target": k // 6}
        for k in range(12)
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
    for column, predictors in (("flag", list(GIVEN)), ("dose", [*GIVEN, "flag"])):
        oracle = DecisionTreeClassifier(min_samples_leaf=5, random_state=0)
        oracle.fit(train[predictors].to_numpy(float), train[column])
        ours = oracle.apply(rows[predictors].to_numpy(float))
        theirs = oracle.apply(train.loc[donors[column], predictors].to_numpy(float))
        assert (ours == theirs).all()
        assert rows[column].tolist() == train.loc[donors[column], column].tolist()
    assert rows["flag"].tolist() == drawn["sex"].tolist()
    assert rows["dose"].tolist() == (2 * drawn["sex"] + drawn["target"]).tolist()
    assert set(donors.columns) == set(VISIT)
    assert donors.isin(train.index).all().all()


def test_the_given_columns_are_untouched():
    drawn = _drawn()
    rows, _ = _redrawn(drawn)
    pd.testing.assert_frame_equal(rows[list(GIVEN)], drawn[list(GIVEN)])
    assert not rows[list(VISIT)].equals(drawn[list(VISIT)])
    assert list(rows.columns) == list(PAIRED.columns)


def test_changing_a_given_value_changes_the_leaf():
    drawn = _drawn()
    flipped = drawn.assign(sex=1 - drawn["sex"])
    rows, _ = _redrawn(drawn)
    moved, _ = _redrawn(flipped)
    assert rows["flag"].tolist() == [k % 2 for k in range(12)]
    assert moved["flag"].tolist() == [1 - k % 2 for k in range(12)]


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


@pytest.mark.parametrize(
    ("given", "visit"),
    [((), VISIT), (GIVEN, ("sex", "dose")), (GIVEN, ("flag", "weight"))],
    ids=["no-given", "overlap", "outside"],
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
