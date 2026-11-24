"""The label bootstrap: whole label-group records drawn into the cell."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.baselines.bootstrap import label_bootstrap
from bko.baselines.errors import BaselineError
from bko.baselines.records import derived_seed
from bko.baselines.request import Request, Strata
from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import EDGE, EVALUATION_COLUMNS, QUOTA, assert_contract, full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()
SEXES = {"female": 0, "male": 1}
STRATUM = (f">={EDGE}", "female", 1)


def pool(train: pd.DataFrame, stratum: tuple[str | int, ...]) -> pd.DataFrame:
    """The label group's records in the stratum's band, banded by hand."""
    inside = train[LABEL] == stratum[-1]
    if len(stratum) == 3:
        old = train["age"] >= EDGE
        inside &= old if stratum[0] == f">={EDGE}" else ~old
    return train.loc[inside]


def test_every_added_row_is_a_label_group_record_in_the_cells_band_except_for_sex():
    request = Request(quota=QUOTA, frame=UCI_FRAME)
    added = label_bootstrap(TRAIN, request).added
    assert_contract(added, TRAIN, QUOTA)
    start = 0
    for stratum, rows in request.quota.items():
        part = added.iloc[start : start + rows]
        start += rows
        held = pool(TRAIN, stratum).drop(columns="sex")
        held_rows = {tuple(row) for row in held.itertuples(index=False)}
        got = {tuple(row) for row in part.drop(columns="sex").itertuples(index=False)}
        assert got <= held_rows
        assert (part["sex"] == SEXES[str(stratum[1])]).all()


def test_the_draw_is_uniform_with_replacement_at_the_derived_seed():
    request = Request(quota=QUOTA, frame=UCI_FRAME)
    added = label_bootstrap(TRAIN, request).added
    start = 0
    for stratum, rows in request.quota.items():
        held = pool(TRAIN, stratum)
        rng = np.random.default_rng(derived_seed(request.seed, "label-bootstrap", *stratum))
        expected = held.iloc[rng.integers(len(held), size=rows)].reset_index(drop=True)
        expected["sex"] = SEXES[str(stratum[1])]
        got = added.iloc[start : start + rows].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected.loc[:, EVALUATION_COLUMNS])
        start += rows


def test_the_same_seed_gives_equal_rows_twice():
    request = Request(quota=QUOTA, frame=UCI_FRAME)
    pd.testing.assert_frame_equal(
        label_bootstrap(TRAIN, request).added, label_bootstrap(TRAIN, request).added
    )


def test_adding_a_stratum_leaves_the_others_rows_unchanged():
    few = Request(quota={STRATUM: 4}, frame=UCI_FRAME)
    more = Request(quota={STRATUM: 4, (f"<{EDGE}", "male", 0): 3}, frame=UCI_FRAME)
    first = label_bootstrap(TRAIN, few).added
    both = label_bootstrap(TRAIN, more).added
    rest = both.loc[both["age"] >= EDGE].loc[both["sex"] == 0]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), rest.reset_index(drop=True))


def test_a_stratum_with_no_label_record_in_its_band_raises_baseline_error():
    train = TRAIN.loc[~((TRAIN["age"] >= EDGE) & (TRAIN[LABEL] == 1))]
    with pytest.raises(BaselineError, match="female"):
        label_bootstrap(train, Request(quota={STRATUM: 2}, frame=UCI_FRAME))


def test_a_label_stratum_draws_any_record_of_the_label_unchanged():
    request = Request(quota={(1,): 6}, strata=Strata.LABEL, frame=UCI_FRAME)
    added = label_bootstrap(TRAIN, request).added
    held = {tuple(row) for row in pool(TRAIN, (1,)).itertuples(index=False)}
    assert {tuple(row) for row in added.itertuples(index=False)} <= held
    assert len(added) == 6


def test_the_pool_sizes_are_logged_as_params():
    result = label_bootstrap(TRAIN, Request(quota=QUOTA, frame=UCI_FRAME))
    assert result.params == {
        "pool": {"/".join(map(str, stratum)): len(pool(TRAIN, stratum)) for stratum in QUOTA}
    }
