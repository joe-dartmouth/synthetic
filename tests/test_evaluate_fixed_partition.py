"""The fixed-partition seam: a dataset with held ids scores the same test records at every seed.

`bko.datasets.split.fixed` holds a published partition's test records out, and
`bko.evaluate.data.held_split` chooses it over the seed-based `split` when the data holds ids
(ADR `research/tableshift-comparison`, Decision 7).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from bko.baselines.request import Request
from bko.datasets.diabetes import parse_diabetic
from bko.datasets.errors import ContractError
from bko.datasets.split import fixed, split
from bko.datasets.tableshift import kept_rows, partition
from bko.datasets.tableshift_frame import encode, tableshift_frame
from bko.evaluate.data import Data, Dataset, held_split
from bko.evaluate.direct import DirectArm
from bko.evaluate.experiment import DIRECT_ARM, E1, gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.runs_published import P1
from bko.ontology.frames import LAB_FIELD_ROWS
from tests._diabetes_fixtures import text
from tests._frame_fixtures import TOY, toy_records
from tests._tableshift_fixtures import tableshift_lines


def _tableshift() -> Data:
    raw = parse_diabetic(text(*tableshift_lines()))
    kept = kept_rows(raw)
    parts = partition(raw)
    records = encode(kept).loc[sorted([*parts.train, *parts.test])]
    held = frozenset(int(record) for record in parts.test)
    return Data(records, tableshift_frame(kept), LAB_FIELD_ROWS, "toy", None, held=held)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_a_fixed_partition_gives_the_same_test_ids_at_every_seed(seed):
    data = _tableshift()
    assert data.held is not None
    parts = held_split(data, seed)
    assert set(parts.test.index) == data.held
    assert set(parts.train.index) == set(data.records.index) - data.held
    assert parts.seed == seed
    assert parts.fingerprint == held_split(data, 0).fingerprint


@pytest.mark.parametrize("seed", [0, 3])
def test_a_dataset_without_held_ids_splits_as_before(seed):
    data = Data(toy_records(), TOY, (), "toy", None)
    assert data.held is None
    expected = split(data.records, seed=seed, edges=TOY.edges)
    found = held_split(data, seed)
    pd.testing.assert_frame_equal(found.train, expected.train)
    pd.testing.assert_frame_equal(found.test, expected.test)
    assert found.seed == seed


def test_a_held_id_missing_from_the_records_is_refused():
    records = toy_records()
    missing = int(records.index.max()) + 1
    with pytest.raises(ContractError, match=f"held record {missing}"):
        fixed(records, held=frozenset({int(records.index[0]), missing}), seed=0)


def test_bko_direct_draws_on_the_tableshift_fixture():
    data = _tableshift()
    train = held_split(data, 0).train
    request: Request = gap_request(train, E1, seed=0, frame=data.frame)
    arm = DirectArm(LAB_FIELD_ROWS, {}, name=DIRECT_ARM, budget=3, sink=None, dataset="toy")
    added = arm(train, request).added
    assert len(added) == sum(request.quota.values())
    assert tuple(added.columns) == data.frame.columns


def test_p1_names_the_stated_configuration():
    arms = (
        "none",
        "class-weight",
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "gaussian-copula",
        "cart",
        "bayes-net",
        "tabddpm",
        "label-bootstrap",
        "bko-direct",
    )
    assert P1.arms == arms
    assert P1.holdout_seeds == tuple(range(10))
    assert P1.candidate == "bko-direct"
    assert P1.comparators == arms[:-1]
    assert P1.replica_arms == (
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "gaussian-copula",
        "cart",
        "bayes-net",
        "tabddpm",
        "label-bootstrap",
        "bko-direct",
    )
    assert P1.criteria == ()
    assert P1.decontaminate is False
    assert P1.removal is None
    assert (
        dataclasses.replace(
            P1,
            holdout_seeds=E1.holdout_seeds,
            arms=E1.arms,
            candidate=E1.candidate,
            comparators=E1.comparators,
            replica_arms=E1.replica_arms,
            criteria=E1.criteria,
            decontaminate=E1.decontaminate,
        )
        == E1
    )
    assert RUNS["p1-id"].experiment == P1
    assert RUNS["p1-ood"].experiment == P1
    assert RUNS["p1-id"].dataset is Dataset.TABLESHIFT_ID
    assert RUNS["p1-ood"].dataset is Dataset.TABLESHIFT_OOD
    assert list(RUNS)[-2:] == ["p1-id", "p1-ood"]
