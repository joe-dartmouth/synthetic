"""SMOTE and ADASYN on a full-feature frame, with real imbalanced-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.oversample import adasyn, smote
from bko.baselines.request import Request
from tests._baselines_fixtures import EDGE, QUOTA, assert_contract, full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()
YOUNG = (TRAIN["age"] < EDGE) & (TRAIN["sex"] == 0) & (TRAIN["target"] == 1)
OLD = (TRAIN["age"] >= EDGE) & (TRAIN["sex"] == 0) & (TRAIN["target"] == 1)
OVERSAMPLERS = {"smote": smote, "adasyn": adasyn}
MEASURED = ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(params=sorted(OVERSAMPLERS))
def oversampler(request):
    return OVERSAMPLERS[request.param]


def test_a_method_meets_the_contract(oversampler):
    assert_contract(oversampler(TRAIN, Request(quota=QUOTA)).added, TRAIN, QUOTA)


def test_the_input_frame_is_not_modified(oversampler):
    before = TRAIN.copy(deep=True)
    oversampler(TRAIN, Request(quota=QUOTA))
    pd.testing.assert_frame_equal(TRAIN, before)


def test_the_seed_alone_decides_the_rows(oversampler):
    np.random.seed(1)
    first = oversampler(TRAIN, Request(quota=QUOTA, seed=3)).added
    np.random.seed(2)
    second = oversampler(TRAIN, Request(quota=QUOTA, seed=3)).added
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_different_rows(oversampler):
    """The positive control: an adapter that ignores the seed passes the test above."""
    first = oversampler(TRAIN, Request(quota=QUOTA, seed=3)).added
    assert not first.equals(oversampler(TRAIN, Request(quota=QUOTA, seed=4)).added)


def test_a_two_record_stratum_uses_one_neighbour(oversampler):
    ids = TRAIN.index[YOUNG]
    train = TRAIN.drop(ids[1:-1])
    kept = train.loc[[ids[0], ids[-1]], MEASURED]
    result = oversampler(train, Request(quota={(f"<{EDGE}", "female", 1): 4}))
    assert len(result.added) == 4
    for name in MEASURED:
        assert result.added[name].between(kept[name].min(), kept[name].max()).all(), name
    assert result.params["neighbors"] == {f"('<{EDGE}', 'female', 1)": 1}


def test_a_one_record_stratum_is_refused(oversampler):
    train = TRAIN.drop(TRAIN.index[YOUNG][1:])
    with pytest.raises(BaselineError, match="at least 2 training records"):
        oversampler(train, Request(quota={(f"<{EDGE}", "female", 1): 4}))


@pytest.mark.parametrize("rows", [1, 2, 17])
def test_adasyn_meets_the_quota_exactly(rows):
    result = adasyn(TRAIN, Request(quota={(f">={EDGE}", "female", 1): rows}))
    assert len(result.added) == rows


def test_adasyn_takes_each_code_from_one_parent():
    train = TRAIN.copy()
    train.loc[OLD, "cp"] = [1, 4] * 7
    result = adasyn(train, Request(quota={(f">={EDGE}", "female", 1): 10}))
    assert set(result.added["cp"]) <= {1, 4}


class Refusing:
    """An ADASYN that refuses, as imbalanced-learn does when no neighbour is in the majority."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit_resample(self, _x: object, _y: object) -> object:
        msg = "Not any neigbours belong to the majority class."
        raise RuntimeError(msg)


def test_adasyn_reports_a_library_refusal(monkeypatch):
    monkeypatch.setattr("bko.baselines.oversample.ADASYN", Refusing)
    with pytest.raises(BaselineError, match="adasyn cannot fill"):
        adasyn(TRAIN, Request(quota=QUOTA))
