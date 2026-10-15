"""The output contract: rounding, numbering, seeds, and the check every method's output passes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.baselines.errors import OutputContractError
from bko.baselines.records import (
    FRAME_COLUMNS,
    FRAME_DTYPES,
    Baseline,
    check,
    derived_seed,
    keep,
    number,
    unit_weights,
    whole,
)
from bko.baselines.request import Request
from bko.datasets.cells import LABEL
from tests._baselines_fixtures import EDGE, QUOTA, cell_list, full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()
REQUEST = Request(quota=QUOTA)

DERIVED = 1368430203
"""`derived_seed(0, "smote", "<55", "female", 1)` at the commit this test was written."""

KEPT = [2, 5, 9]
"""`keep(3, 10, seed=0, label="adasyn")` at the same commit."""


def valid_rows() -> pd.DataFrame:
    """The quota's rows, taken from the training frame itself, numbered as added rows."""
    cells = cell_list(TRAIN)
    parts = [
        TRAIN[[cell == stratum for cell in cells]].head(rows)
        for stratum, rows in REQUEST.quota.items()
    ]
    return number(pd.concat(parts))


def test_the_fixture_puts_every_code_in_every_cell():
    for _, cell in TRAIN.groupby([TRAIN["age"] >= EDGE, "sex", LABEL]):
        assert set(cell["thal"]) == {3, 6, 7}
        assert set(cell["cp"]) == {1, 2, 3, 4}


def one_row(**values: float) -> pd.DataFrame:
    row = TRAIN.iloc[[0]].astype("float64")
    for name, value in values.items():
        row[name] = value
    return row


def test_whole_rounds_rather_than_truncates():
    out = whole(one_row(age=54.6, thal=6.6, oldpeak=1.26))
    assert (out["age"].item(), out["thal"].item(), out["oldpeak"].item()) == (55, 7, 1.3)
    assert out.dtypes.astype(str).to_dict() == dict(FRAME_DTYPES)


def test_whole_refuses_a_missing_value():
    with pytest.raises(OutputContractError, match="missing or infinite"):
        whole(one_row(chol=np.nan))


def test_number_indexes_from_minus_one():
    out = number(TRAIN.head(3))
    assert out.index.tolist() == [-1, -2, -3]
    assert out.index.name == "record_id"


def test_a_valid_output_passes():
    baseline = Baseline(added=valid_rows(), weights=unit_weights(TRAIN), params={})
    check(baseline, TRAIN, REQUEST)


def _set_last(frame: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    out = frame.copy()
    out.loc[out.index[-1], column] = value
    return out


def _flip_sex(frame: pd.DataFrame) -> pd.DataFrame:
    return _set_last(frame, "sex", 1 - int(frame["sex"].iloc[-1]))


BROKEN_ROWS = {
    "short": lambda rows: rows.iloc[:-1],
    "columns": lambda rows: rows.drop(columns="chol"),
    "dtype": lambda rows: rows.astype({"chol": "float64"}),
    "index": lambda rows: rows.reset_index(drop=True),
    "codebook": lambda rows: _set_last(rows, "thal", 4),
    "label": lambda rows: _set_last(rows, LABEL, 2),
    "range": lambda rows: _set_last(rows, "chol", int(TRAIN["chol"].max()) + 1),
    "decimals": lambda rows: _set_last(rows, "oldpeak", float(rows["oldpeak"].iloc[-1]) + 0.05),
    "stratum": _flip_sex,
    "missing": lambda rows: _set_last(rows, "oldpeak", float("nan")),
    "below range": lambda rows: _set_last(rows, "chol", int(TRAIN["chol"].min()) - 1),
    "index name": lambda rows: rows.rename_axis("id"),
    "column order": lambda rows: rows.loc[:, [*rows.columns[1:], rows.columns[0]]],
    "strata order": lambda rows: rows.iloc[::-1].set_axis(rows.index),
}
"""Each plants one broken promise in an otherwise valid added frame."""


@pytest.mark.parametrize("plant", BROKEN_ROWS.values(), ids=list(BROKEN_ROWS))
def test_each_broken_row_promise_is_refused(plant):
    baseline = Baseline(added=plant(valid_rows()), weights=unit_weights(TRAIN), params={})
    with pytest.raises(OutputContractError, match="breaks its contract"):
        check(baseline, TRAIN, REQUEST)


BROKEN_WEIGHTS = {
    "weights index": lambda weights: weights.set_axis(weights.index + 1000),
    "weight": lambda weights: weights.where(weights.index != weights.index[0], 0.0),
    "negative weight": lambda weights: weights.where(weights.index != weights.index[0], -1.0),
    "missing weight": lambda weights: weights.where(weights.index != weights.index[0], np.nan),
}
"""Each plants one broken promise in otherwise valid weights."""


@pytest.mark.parametrize("plant", BROKEN_WEIGHTS.values(), ids=list(BROKEN_WEIGHTS))
def test_each_broken_weight_promise_is_refused(plant):
    baseline = Baseline(added=valid_rows(), weights=plant(unit_weights(TRAIN)), params={})
    with pytest.raises(OutputContractError, match="breaks its contract"):
        check(baseline, TRAIN, REQUEST)


def test_the_frame_columns_are_the_evaluation_frames():
    assert list(FRAME_COLUMNS) == list(TRAIN.columns)


def test_derived_seeds_differ_by_part_and_stay_in_range():
    seeds = [
        derived_seed(0, "smote", "<55"),
        derived_seed(0, "smote", ">=55"),
        derived_seed(1, "smote", "<55"),
    ]
    assert len(set(seeds)) == 3
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert derived_seed(0, "smote", "<55") == seeds[0]


def test_keep_is_a_sorted_seeded_subset():
    kept = keep(3, 10, seed=0, label="x")
    assert len(set(kept)) == 3
    assert kept == sorted(kept)
    assert set(kept) <= set(range(10))
    assert keep(3, 10, seed=0, label="x") == kept
    labels = [f"x{n}" for n in range(25)]
    assert any(
        keep(3, 10, seed=0, label=label) != keep(3, 10, seed=1, label=label) for label in labels
    )


def test_derived_seeds_and_kept_positions_are_pinned():
    """Golden values: a run reproduces across commits only while these stay fixed."""
    assert derived_seed(0, "smote", "<55", "female", 1) == DERIVED
    assert keep(3, 10, seed=0, label="adasyn") == KEPT
