"""The field-group match attack and the donor reuse of `python -m bko.evaluate privacy`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.donor_attacks import MATCH_MEASURES, match_groups, match_measures, match_scores
from bko.evaluate.scores import Measure
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
)
from tests.test_evaluate_privacy import random_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CELL = ("<55", "female", 1)
"""A cell the planted records may fall in."""

UNITS = tuple((name,) for name in UCI_FRAME.columns if name not in {"age", "sex", "target"})
"""UCI's columns as singletons, but `age`, which places a record's cell, `sex` and the label."""


def _planted(count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Members, and non-members that share no unit value with them: every coded unit at another
    code and every measured unit out of the members' range."""
    members = random_records(UCI_FRAME, count, seed=seed)
    nonmembers = random_records(UCI_FRAME, count, seed=seed + 1)
    for (name,) in UNITS:
        if name in UCI_FRAME.numeric:
            nonmembers[name] += 10_000
        else:
            first, second = list(UCI_FRAME.codebook[name])[:2]
            members[name], nonmembers[name] = first, second
    return members, nonmembers


def _measures(added: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return match_measures(added, train=train, test=test, cell=CELL, frame=UCI_FRAME, groups=UNITS)


def test_the_match_score_is_the_most_units_one_added_row_copies():
    records = pd.DataFrame({"a": [1, 5, 9], "b": [1, 6, 9], "c": [1, 7, 9], "d": [3, 0, 0]})
    added = pd.DataFrame({"a": [1, 5, 0], "b": [1, 6, 0], "c": [1, 0, 7], "d": [3, 0, 3]})
    groups = (("a",), ("b", "d"), ("c",))
    found = match_scores(records, added, groups)
    assert found.dtype.kind == "i"
    assert found.tolist() == [3, 2, 0]


def test_a_planted_verbatim_copy_scores_auc_1():
    members, nonmembers = _planted(300, seed=21)
    found = _measures(members.copy(), members, nonmembers)
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(1.0, abs=0)
    assert found[Measure.MATCH_TPR01_OUTSIDE] == pytest.approx(1.0, abs=0)
    assert found[Measure.MATCH_AUC_LABEL] == pytest.approx(1.0, abs=0)


def test_disjoint_rows_score_one_half():
    members, nonmembers = _planted(300, seed=31)
    added = nonmembers.copy()
    for (name,) in UNITS:
        added[name] = added[name] + 100_000 if name in UCI_FRAME.numeric else -1
    assert not match_scores(members, added, UNITS).any()
    assert not match_scores(nonmembers, added, UNITS).any()
    found = _measures(added, members, nonmembers)
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(0.5, abs=0)
    assert found[Measure.MATCH_AUC_LABEL] == pytest.approx(0.5, abs=0)


def test_partial_copies_read_the_hand_auc():
    members, nonmembers = _planted(2, seed=41)
    for frame in (members, nonmembers):
        frame["age"], frame["sex"], frame["target"] = 70, 1, 1
    groups = (("cp",), ("trestbps",), ("chol",))
    copied = nonmembers.iloc[[0]].copy()
    copied["cp"], copied["chol"] = -1, -1
    added = pd.concat([members.iloc[[0]], copied], ignore_index=True)
    assert match_scores(members, added, groups).tolist() == [3, 1]
    assert match_scores(nonmembers, added, groups).tolist() == [1, 0]
    found = match_measures(
        added, train=members, test=nonmembers, cell=CELL, frame=UCI_FRAME, groups=groups
    )
    # Members [3, 1] against non-members [1, 0]: three wins of four pairs and one tie.
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(0.875, abs=0)


def test_an_arm_with_no_rows_reads_nan():
    members, nonmembers = _planted(20, seed=51)
    found = _measures(members.iloc[:0], members, nonmembers)
    assert list(found) == list(MATCH_MEASURES)
    assert all(math.isnan(value) for value in found.values())


@pytest.mark.parametrize(
    ("rows", "frame"),
    [(DIABETES_FIELD_ROWS, DIABETES_FRAME), (SYNTHEA_FIELD_ROWS, SYNTHEA_FRAME), ((), UCI_FRAME)],
)
def test_every_column_but_sex_and_the_label_is_in_exactly_one_unit(rows, frame):
    groups = match_groups(rows, frame=frame)
    named = [name for group in groups for name in group]
    assert sorted(named) == sorted(set(named))
    assert set(named) == set(frame.columns) - {"sex", frame.label}
    firsts = [frame.columns.index(group[0]) for group in groups]
    assert firsts == sorted(firsts)


def test_the_match_measures_mirror_the_distance_attacks_scopes():
    assert [m.value for m in MATCH_MEASURES] == [
        "match-auc-outside",
        "match-tpr01-outside",
        "match-tpr001-outside",
        "match-auc-cell",
        "match-tpr01-cell",
        "match-auc-label",
        "match-tpr01-label",
        "match-tpr001-label",
    ]
    assert np.array_equal(
        match_scores(pd.DataFrame({"a": [np.nan]}), pd.DataFrame({"a": [np.nan]}), (("a",),)),
        [1],
    )
