"""`bko.evaluate.curation`: CLLM's curator drops a row the classifier confidently puts elsewhere."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.curation import classifier_discard, discards
from tests._baselines_fixtures import full_records


def separable() -> pd.DataFrame:
    """`full_records` whose label `thalach` separates: low for the sick, high for the well."""
    train = full_records().reset_index(drop=True)
    rng = np.random.default_rng(1)
    sick = train[LABEL] == 1
    train.loc[sick, "thalach"] = rng.integers(80, 121, int(sick.sum()))
    train.loc[~sick, "thalach"] = rng.integers(160, 201, int((~sick).sum()))
    return train


def test_the_curator_drops_a_confident_wrong_row():
    train = separable()
    discard = classifier_discard(train, frame=UCI_FRAME)
    rows = train.loc[[0, 0, 1]].reset_index(drop=True)
    labels = rows[LABEL].to_list()
    rows.loc[1, "thalach"] = 190 if labels[1] == 1 else 90
    rows.loc[2, "thalach"] = 100 if labels[2] == 1 else 180
    dropped = discard(rows)
    assert dropped.tolist() == [False, True, False]
    assert dropped.index.equals(rows.index)


def test_the_curator_keeps_an_exact_set_at_one_seed():
    train = separable()
    pool = train.sample(12, random_state=3).reset_index(drop=True)
    pool.loc[[2, 7], "thalach"] = pool.loc[[2, 7], LABEL].map({1: 195, 0: 85})
    dropped = classifier_discard(train, frame=UCI_FRAME)(pool)
    assert tuple(np.flatnonzero(dropped.to_numpy())) == (2, 7)
    assert dropped.equals(classifier_discard(train, frame=UCI_FRAME)(pool))


@pytest.mark.parametrize(
    ("confidence", "aleatoric", "dropped"),
    [(0.3, 0.1, False), (0.2, 0.15, True), (0.1, 0.2, False), (0.05, 0.04, True)],
    ids=["aleatoric-alone", "both-at-threshold", "uncertain", "confidently-wrong"],
)
def test_aleatoric_alone_does_not_drop(confidence, aleatoric, dropped):
    assert discards(np.array([confidence]), np.array([aleatoric])).tolist() == [dropped]


def test_one_label_is_refused():
    train = separable()
    with pytest.raises(ValueError, match="both labels"):
        classifier_discard(train[train[LABEL] == 1], frame=UCI_FRAME)


def test_no_rows_drop_nothing():
    found = classifier_discard(separable(), frame=UCI_FRAME)(separable().head(0))
    assert found.empty
