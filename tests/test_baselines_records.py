"""The output contract: rounding, numbering, seeds, and the check every method's output passes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bko.baselines.errors import OutputContractError
from bko.baselines.records import (
    Baseline,
    Drop,
    check,
    derived_seed,
    drops,
    keep,
    number,
    training_set,
    unit_weights,
    whole,
)
from bko.baselines.request import Request
from bko.datasets.cells import LABEL
from bko.datasets.uci import FRAME_COLUMNS, FRAME_DTYPES, NUMERIC, UCI_FRAME
from bko.validate.checks import validate
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


def test_training_set_puts_the_real_rows_first_with_their_weights():
    added = valid_rows()
    weights = unit_weights(TRAIN).mul(2.0)
    frame = training_set(TRAIN, Baseline(added=added, weights=weights, params={}))
    assert list(frame.columns) == [*FRAME_COLUMNS, "weight"]
    assert frame.index.tolist() == [*TRAIN.index, *added.index]
    assert frame["weight"].tolist() == [2.0] * len(TRAIN) + [1.0] * len(added)


@pytest.mark.parametrize("plant", ["range", "below range"])
def test_a_value_outside_the_split_names_its_column(plant):
    baseline = Baseline(
        added=BROKEN_ROWS[plant](valid_rows()), weights=unit_weights(TRAIN), params={}
    )
    with pytest.raises(OutputContractError, match="chol leaves"):
        check(baseline, TRAIN, REQUEST)


def test_a_row_past_its_quota_is_refused():
    rows = valid_rows()
    added = number(pd.concat([rows, rows.iloc[[-1]]]))
    baseline = Baseline(added=added, weights=unit_weights(TRAIN), params={})
    with pytest.raises(OutputContractError, match="rows per stratum"):
        check(baseline, TRAIN, REQUEST)


def candidates_of(rows: pd.DataFrame) -> list[dict[str, object]]:
    """`rows` as decoded candidates: JSON numbers, as the API returns them."""
    return json.loads(rows.to_json(orient="records"))


def test_a_row_out_of_range_is_dropped_for_range():
    rows = _set_last(valid_rows(), "chol", int(TRAIN["chol"].max()) + 1)
    assert drops(rows, TRAIN, REQUEST)[-1] is Drop.RANGE


def test_a_row_of_an_unrequested_cell_is_dropped_for_its_cell():
    other = TRAIN[[cell not in REQUEST.quota for cell in cell_list(TRAIN)]].iloc[[0]]
    rows = number(pd.concat([valid_rows(), other]))
    assert drops(rows, TRAIN, REQUEST)[-1] is Drop.CELL


def test_a_row_past_its_cells_quota_is_dropped_for_quota():
    rows = valid_rows()
    rows = number(pd.concat([rows, rows.iloc[[-1]]]))
    assert drops(rows, TRAIN, REQUEST) == (*(None for _ in range(len(rows) - 1)), Drop.QUOTA)


def test_held_rows_use_up_the_quota():
    first = valid_rows().iloc[[0]]
    stratum = cell_list(first)[0]
    assert drops(first, TRAIN, REQUEST, held={stratum: REQUEST.quota[stratum]}) == (Drop.QUOTA,)


def test_a_dropped_row_uses_none_of_the_quota():
    first = valid_rows().iloc[[0]]
    outside = _set_last(first.copy(), "chol", int(TRAIN["chol"].max()) + 1)
    assert drops(number(pd.concat([outside, first])), TRAIN, REQUEST) == (Drop.RANGE, None)


def test_a_valid_frame_drops_nothing():
    assert set(drops(valid_rows(), TRAIN, REQUEST)) == {None}


def test_check_and_drops_share_the_range_rule(monkeypatch):
    rows = _set_last(valid_rows(), "chol", int(TRAIN["chol"].max()) + 1)

    def everything(frame: pd.DataFrame, train: pd.DataFrame, _spec: object) -> pd.DataFrame:
        return pd.DataFrame(True, index=frame.index, columns=list(NUMERIC))

    monkeypatch.setattr("bko.baselines.records.within_split", everything)
    assert drops(rows, TRAIN, REQUEST)[-1] is None
    check(Baseline(added=rows, weights=unit_weights(TRAIN), params={}), TRAIN, REQUEST)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("age", 18),
        ("age", 120),
        ("chol", 32),
        ("chol", 600),
        ("oldpeak", -10.0),
        ("oldpeak", 10.0),
        ("sex", None),
        ("chol", None),
    ],
    ids=[
        "age-18",
        "age-120",
        "chol-32",
        "chol-600",
        "oldpeak-low",
        "oldpeak-high",
        "other-cell",
        "past-quota",
    ],
)
def test_validated_candidates_leave_check_nothing_to_refuse(name, value):
    rows = valid_rows()
    extra = candidates_of(rows.iloc[[-1]])[0]
    if name == "sex":
        extra["sex"] = 1 - int(str(extra["sex"]))
    elif value is not None:
        extra[name] = value
    accepted = validate([*candidates_of(rows), extra], frame=UCI_FRAME).accepted
    reasons = drops(accepted, TRAIN, REQUEST)
    assert reasons[-1] is not None
    assert set(reasons[:-1]) == {None}
    kept = number(accepted.loc[[reason is None for reason in reasons]])
    check(Baseline(added=kept, weights=unit_weights(TRAIN), params={}), TRAIN, REQUEST)
