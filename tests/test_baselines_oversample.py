"""SMOTE and ADASYN on a full-feature frame, with real imbalanced-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.baselines.errors import BaselineError, ShortfallError
from bko.baselines.oversample import adasyn, smote
from bko.baselines.request import Request
from bko.evaluate.data import digest
from tests._baselines_fixtures import (
    EDGE,
    EVALUATION_COLUMNS,
    QUOTA,
    assert_contract,
    full_records,
)


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
    result = oversampler(TRAIN, Request(quota=QUOTA))
    assert_contract(result.added, TRAIN, QUOTA)
    assert set(result.params["neighbors"].values()) == {5}  # 14 records: the cap of 5 applies


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


@pytest.mark.parametrize("left", [0, 1])
def test_a_stratum_of_fewer_than_two_records_raises_a_shortfall_holding_the_others(
    oversampler, left
):
    train = TRAIN.drop(TRAIN.index[YOUNG][left:])
    young = (f"<{EDGE}", "female", 1)
    with pytest.raises(ShortfallError, match="at least 2 training records") as caught:
        oversampler(train, Request(quota=QUOTA, seed=3))
    short = caught.value
    assert short.short == {f"<{EDGE}/female/1": QUOTA[young]}
    others = {stratum: rows for stratum, rows in QUOTA.items() if stratum != young}
    assert short.baseline is not None
    assert_contract(short.baseline.added, train, others)
    pd.testing.assert_frame_equal(short.added, short.baseline.added)
    assert short.baseline.params == {
        "neighbors": {str(stratum): 5 for stratum in others},
        "short": {f"<{EDGE}/female/1": QUOTA[young]},
    }


def test_every_stratum_empty_raises_a_shortfall_of_no_rows(oversampler):
    train = TRAIN.drop(TRAIN.index[YOUNG])
    with pytest.raises(ShortfallError) as caught:
        oversampler(train, Request(quota={(f"<{EDGE}", "female", 1): 4}))
    assert caught.value.baseline is not None
    assert caught.value.baseline.added.empty
    assert list(caught.value.baseline.added.columns) == EVALUATION_COLUMNS


PINNED = {
    ("smote", 0): "4816f15feb6fac53cdf9a3188225f25853488ecc8b8f091d23ed46648d703e5c",
    ("smote", 3): "6e28dec998722bacaffb007062c3617feb681b06d1de873c20912b06ec59ce5a",
    ("adasyn", 0): "0fa5ece66778041539e05c3cbfaeb6e0704416dfba33e94bbadf8fa8f5de857d",
    ("adasyn", 3): "281ced36241feb47b9a1bb1e51a1891fe5f817627b98cfa10a39cc46afd3cf47",
}
"""Each method's added rows' `digest` on `QUOTA`, as the method wrote them before a shortfall."""


@pytest.mark.parametrize(("method", "seed"), sorted(PINNED))
def test_smote_and_adasyn_bytes_are_unchanged_off_the_structural_path(method, seed):
    result = OVERSAMPLERS[method](TRAIN, Request(quota=QUOTA, seed=seed))
    assert digest(result.added) == PINNED[method, seed]
    assert result.params == {"neighbors": {str(stratum): 5 for stratum in sorted(QUOTA)}}


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
    """A sampler that refuses, as imbalanced-learn does when it cannot fill a class."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit_resample(self, _x: object, _y: object) -> object:
        msg = "Not any neigbours belong to the majority class."
        raise RuntimeError(msg)


def test_adasyn_reports_a_library_refusal(monkeypatch):
    monkeypatch.setattr("bko.baselines.oversample.ADASYN", Refusing)
    with pytest.raises(BaselineError, match="adasyn cannot fill"):
        adasyn(TRAIN, Request(quota=QUOTA))


def test_smote_reports_a_library_refusal(monkeypatch):
    monkeypatch.setattr("bko.baselines.oversample.SMOTENC", Refusing)
    with pytest.raises(BaselineError, match="smote cannot fill"):
        smote(TRAIN, Request(quota=QUOTA))
