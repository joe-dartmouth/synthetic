"""`augment` and the controls: refusals, the contract check, and the methods that add no rows."""

from __future__ import annotations

from functools import partial

import pandas as pd
import pytest

from bko.baselines.controls import no_augmentation
from bko.baselines.errors import BaselineError, OutputContractError
from bko.baselines.methods import METHODS, augment
from bko.baselines.oversample import adasyn, smote
from bko.baselines.records import unit_weights
from bko.baselines.request import Method, Request, Strata
from bko.baselines.synthesizers import ctgan, tvae
from tests._baselines_fixtures import (
    EVALUATION_COLUMNS,
    EVALUATION_DTYPES,
    QUOTA,
    TINY_CTGAN,
    TINY_TVAE,
    cells_of,
    full_records,
)


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()


def test_no_augmentation_adds_nothing_and_weights_one():
    result = augment(TRAIN, Method.NONE, Request())
    assert result.added.empty
    assert list(result.added.columns) == EVALUATION_COLUMNS
    assert result.added.dtypes.astype(str).to_dict() == EVALUATION_DTYPES
    pd.testing.assert_series_equal(result.weights, unit_weights(TRAIN))


def test_class_weights_follow_n_over_k_n_c():
    small = pd.concat([TRAIN[TRAIN["target"] == 0].head(6), TRAIN[TRAIN["target"] == 1].head(2)])
    result = augment(small, Method.CLASS_WEIGHT, Request())
    weights = result.weights
    assert weights[small["target"] == 0].tolist() == pytest.approx([8 / 12] * 6)
    assert weights[small["target"] == 1].tolist() == pytest.approx([2.0] * 2)
    assert weights.sum() == pytest.approx(8)
    assert result.params["weights"] == {0: pytest.approx(8 / 12), 1: pytest.approx(2.0)}


@pytest.mark.parametrize("control", [Method.NONE, Method.CLASS_WEIGHT])
def test_a_control_refuses_rows(control):
    with pytest.raises(BaselineError, match="needs no rows"):
        augment(TRAIN, control, Request(quota={(1,): 1}, strata=Strata.LABEL))


def test_a_row_adding_method_refuses_an_empty_request():
    with pytest.raises(BaselineError, match="needs at least one row"):
        augment(TRAIN, Method.SMOTE, Request(), methods={Method.SMOTE: no_augmentation})


def test_augment_refuses_a_broken_output():
    with pytest.raises(OutputContractError, match="rows per stratum"):
        augment(TRAIN, Method.SMOTE, Request(quota=QUOTA), methods={Method.SMOTE: no_augmentation})


def test_every_method_has_a_runner():
    assert set(METHODS) == set(Method)


def test_one_request_gives_every_method_the_same_rows_per_cell():
    runners = [
        smote,
        adasyn,
        partial(ctgan, parameters=TINY_CTGAN),
        partial(tvae, parameters=TINY_TVAE),
    ]
    request = Request(quota=QUOTA)
    cells = [cells_of(runner(TRAIN, request).added) for runner in runners]
    assert all(cell == cells[0] for cell in cells)
