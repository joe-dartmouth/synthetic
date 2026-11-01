"""`bko.baselines.bayes_net`: GreedyBayes structure, BDeu tables and donor draws."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bko.baselines.bayes_net import (
    BINS,
    ESS,
    Bins,
    K,
    Network,
    bayes_net,
    discretize,
    draw,
    learn,
    mutual_information,
)
from bko.baselines.records import derived_seed, unit_weights
from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import QUOTA, assert_contract, full_records
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN


def states_of(**columns) -> Bins:
    frame = pd.DataFrame(columns)
    arity = {name: int(frame[name].max()) + 1 for name in frame.columns}
    return Bins({}, {name: tuple(range(n)) for name, n in arity.items()}, frame)


def test_mutual_information_by_hand():
    # a 2x2 table of 40 rows: (0,0) 15, (0,1) 5, (1,0) 5, (1,1) 15
    a = [0] * 20 + [1] * 20
    b = [0] * 15 + [1] * 5 + [0] * 5 + [1] * 15
    found = mutual_information(pd.DataFrame({"a": a, "b": b}), "b", ["a"])
    by_hand = 2 * (15 / 40) * math.log(15 * 40 / (20 * 20)) + 2 * (5 / 40) * math.log(
        5 * 40 / (20 * 20)
    )
    assert found == pytest.approx(by_hand)


def planted_chain(rows: int = 2000) -> Bins:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, rows)
    b = a ^ (rng.random(rows) < 0.05)
    c = b.copy()
    noise = rng.integers(0, 2, rows)
    return states_of(a=a, b=b.astype(int), c=c.astype(int), noise=noise)


def test_greedy_bayes_recovers_a_planted_chain():
    network = learn(planted_chain(), k=1, first="a")
    edges = set(network.edges)
    assert ("a", "b") in edges
    assert ("b", "c") in edges


def test_every_node_gets_min_k_placed_parents():
    network = learn(planted_chain(), k=2, first="noise")
    assert [len(node.parents) for node in network.nodes] == [0, 1, 2, 2]


def test_the_graph_is_acyclic():
    network = learn(planted_chain(), k=2, first="c")
    seen: set[str] = set()
    for node in network.nodes:
        assert set(node.parents) <= seen
        seen.add(node.field)
    assert seen == {"a", "b", "c", "noise"}


def test_bdeu_conditionals_by_hand():
    bins = states_of(a=[0, 0, 0, 1], b=[0, 0, 1, 1])
    network = learn(bins, k=1, first="a")
    (node,) = [n for n in network.nodes if n.field == "b"]
    # q = 2 configurations of a, r = 2 states of b: alpha = ESS / 4 per cell
    alpha = ESS / 4
    assert node.chances((0,)) == pytest.approx(
        ((2 + alpha) / (3 + 2 * alpha), (1 + alpha) / (3 + 2 * alpha))
    )
    assert node.chances((1,)) == pytest.approx(
        (alpha / (1 + 2 * alpha), (1 + alpha) / (1 + 2 * alpha))
    )


def test_an_empty_parent_configuration_gets_the_prior():
    bins = Bins({}, {"a": (0, 1, 2), "b": (0, 1)}, pd.DataFrame({"a": [0, 0, 1], "b": [0, 1, 1]}))
    network = learn(bins, k=1, first="a")
    (node,) = [n for n in network.nodes if n.field == "b"]
    assert node.chances((2,)) == (0.5, 0.5)


TRAIN = full_records(20)


def test_bins_are_non_empty_and_force_the_band_edge():
    bins = discretize(TRAIN, UCI_FRAME)
    for field, cuts in bins.cuts.items():
        counts = np.bincount(bins.states[field], minlength=len(cuts) + 1)
        assert (counts > 0).all(), field
    for edge in UCI_FRAME.edges:
        assert float(edge) in bins.cuts["age"]
    young = bins.states["age"][TRAIN["age"] < UCI_FRAME.edges[0]]
    old = bins.states["age"][TRAIN["age"] >= UCI_FRAME.edges[0]]
    assert set(young).isdisjoint(set(old))
    for field in UCI_FRAME.categorical:
        assert bins.arity(field) == len(UCI_FRAME.codebook[field])


def test_a_donor_stays_in_its_bin():
    bins = discretize(TRAIN, UCI_FRAME)
    network = learn(bins, first="age")
    rows = draw(network, bins, TRAIN, 300, np.random.default_rng(1), UCI_FRAME)
    assert list(rows.columns) == list(UCI_FRAME.columns)
    assert dict(rows.dtypes.astype(str)) == dict(UCI_FRAME.dtypes)
    redrawn = discretize(TRAIN, UCI_FRAME)
    for field in UCI_FRAME.columns:
        assert set(rows[field]) <= set(TRAIN[field]), field
    # every drawn value's state is a state of its own field's donors
    for field, cuts in redrawn.cuts.items():
        states = np.searchsorted(
            np.asarray(cuts), rows[field].to_numpy(dtype="float64"), side="right"
        )
        assert set(states) <= set(redrawn.states[field])


def test_the_same_rng_draws_the_same_rows():
    bins = discretize(TRAIN, UCI_FRAME)
    network = learn(bins, first="sex")
    first = draw(network, bins, TRAIN, 50, np.random.default_rng(4), UCI_FRAME)
    second = draw(network, bins, TRAIN, 50, np.random.default_rng(4), UCI_FRAME)
    assert first.equals(second)
    assert isinstance(network, Network)


def test_an_unknown_first_field_is_refused():
    with pytest.raises(ValueError, match="first field"):
        learn(planted_chain(), first="nowhere")


def _run(ambient: int, seed: int, train: pd.DataFrame = TRAIN, quota=QUOTA):
    np.random.seed(ambient)
    return bayes_net(train, Request(quota=quota, seed=seed))


@pytest.fixture(scope="module")
def runs():
    return _run(1, 3), _run(2, 3), _run(1, 4)


def test_the_network_meets_the_contract(runs):
    assert_contract(runs[0].added, TRAIN, QUOTA)
    pd.testing.assert_series_equal(runs[0].weights, unit_weights(TRAIN))


def test_the_rows_depend_on_the_seed_alone(runs):
    pd.testing.assert_frame_equal(runs[0].added, runs[1].added)
    assert not runs[0].added.equals(runs[2].added)


def test_the_ambient_generator_is_left_alone():
    np.random.seed(9)
    expected = np.random.random()
    np.random.seed(9)
    bayes_net(TRAIN, Request(quota=QUOTA, seed=3))
    assert np.random.random() == expected


def test_the_params_name_the_first_field_and_the_learned_edges(runs):
    params = runs[0].params
    seed = derived_seed(3, "bayes-net")
    assert params["first"] == UCI_FRAME.columns[seed % len(UCI_FRAME.columns)]
    assert (params["k"], params["bins"], params["ess"]) == (K, BINS, ESS)
    network = learn(discretize(TRAIN.reset_index(drop=True), UCI_FRAME), first=params["first"])
    assert params["edges"] == [list(edge) for edge in network.edges]


def test_the_network_fills_the_toy_frame():
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 3)
    request = Request(quota=quota, strata=Strata.CELL, seed=0, frame=TOY)
    added = bayes_net(TOY_TRAIN, request).added
    assert list(added.columns) == list(TOY.columns)
    assert len(added) == sum(quota.values())
