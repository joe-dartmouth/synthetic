"""`bko-tree`'s tree fields and shares on the `diabetes-codes` cohort and the pinned MONDO.

The cohort reads the pinned Diabetes 130 archive and the trees read the pinned releases over its
field rows. It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent
snapshot or release fails it rather than skipping it.
"""

from __future__ import annotations

import functools
import math
from typing import NoReturn

import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.diabetes import DIAGNOSES, load_diabetes
from bko.datasets.fetch import default_data_dir
from bko.datasets.frames import FrameSpec
from bko.datasets.split import split
from bko.evaluate.experiment import gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.runs_diabetes import E6_CELL
from bko.gaps.scope import load_scope
from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.tree import column_trees, leaves, tree_fields, tree_shares
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS, diabetes_codes_frame
from bko.ontology.mapping import RELEASES
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    records = load_diabetes(default_data_dir(), indicators={}, diagnoses=True)
    return records, diabetes_codes_frame(records)


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


@functools.cache
def _ontologies() -> dict[str, Ontology]:
    concepts = {row.concept for row in DIABETES_CODES_FIELD_ROWS}
    ontologies, _ = load_scope(concepts, default_data_dir(), releases=RELEASES, open_url=_no_fetch)
    return dict(ontologies)


def _train(name: str) -> pd.DataFrame:
    records, frame = _loaded()
    parts = split(records.copy(), seed=0, edges=frame.edges)
    return prepared(parts, RUNS[name].experiment, edges=frame.edges).ablated.train


def test_the_tree_fields_of_diabetes_codes_are_the_three_diagnosis_columns():
    """Any other result parks the unit: with no tree field the arm draws as `bko-direct`."""
    _, frame = _loaded()
    assert (
        tree_fields(DIABETES_CODES_FIELD_ROWS, frame) == DIAGNOSES == ("diag_1", "diag_2", "diag_3")
    )


@pytest.mark.parametrize("name", ["e15-thin", "e15-empty"])
def test_each_columns_shares_sum_to_one_in_e6s_cell_and_its_label_group(name):
    _, frame = _loaded()
    train = _train(name)
    inside = cell_keys(train, edges=frame.edges).eq(list(E6_CELL)).all(axis=1).to_numpy()
    group = train.loc[train[frame.label].eq(E6_CELL[2])]
    trees = column_trees(DIABETES_CODES_FIELD_ROWS, frame, _ontologies())
    assert tuple(trees) == DIAGNOSES
    for column, tree in trees.items():
        own = tree.counts(map(int, train.loc[inside, column]))
        held = tree.counts(map(int, group[column]))
        for cell in (own, {}):
            shares, _ = tree_shares(cell, held, tree.tree, GROUP_WEIGHT)
            total = math.fsum(shares[leaf] for leaf in leaves(tree.tree))
            assert math.isclose(total, 1.0, rel_tol=1e-12), (column, total)


@pytest.mark.parametrize(("name", "asked"), [("e15-thin", 1250), ("e15-empty", 1269)])
def test_the_e15_gap_request_asks_e6s_quota(name, asked):
    _, frame = _loaded()
    request = gap_request(
        _train(name), RUNS[name].experiment, seed=0, frame=frame, rows=(), source_name=""
    )
    assert dict(request.quota) == {E6_CELL: asked}
