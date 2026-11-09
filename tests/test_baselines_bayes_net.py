"""`bko.baselines.bayes_net`: GreedyBayes structure, BDeu tables and donor draws."""

from __future__ import annotations

import functools
import hashlib
import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bko.baselines.bayes_net as bayes_net_module
from bko.baselines.bayes_net import (
    BINS,
    ESS,
    Bins,
    K,
    Network,
    Node,
    _occupied,
    _states,
    bayes_net,
    discretize,
    draw,
    learn,
    mutual_information,
)
from bko.baselines.errors import ShortfallError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.records import derived_seed, unit_weights
from bko.baselines.request import Request, Strata, Stratum
from bko.baselines.synthesizers import fill
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import EDGE, QUOTA, assert_contract, full_records
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
    assert (params["seed"], params["k"], params["bins"], params["ess"]) == (3, K, BINS, ESS)
    assert params["states"] == {f: discretize(TRAIN, UCI_FRAME).arity(f) for f in UCI_FRAME.columns}
    network = learn(discretize(TRAIN.reset_index(drop=True), UCI_FRAME), first=params["first"])
    assert params["edges"] == [list(edge) for edge in network.edges]


def test_the_network_fills_the_toy_frame():
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 3)
    request = Request(quota=quota, strata=Strata.CELL, seed=0, frame=TOY)
    added = bayes_net(TOY_TRAIN, request).added
    assert list(added.columns) == list(TOY.columns)
    assert len(added) == sum(quota.values())


def test_the_fit_line_names_the_harness_fill(caplog):
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 3)
    request = Request(quota=quota, strata=Strata.CELL, seed=0, frame=TOY)
    caplog.set_level(logging.INFO, logger=bayes_net_module.logger.name)
    bayes_net(TOY_TRAIN, request)
    token = CURRENT_FILL.set(Fill(2, "gap"))
    try:
        bayes_net(TOY_TRAIN, request)
    finally:
        CURRENT_FILL.reset(token)
    fits = [
        r.getMessage().split(" seconds=")[0]
        for r in caplog.records
        if r.name == bayes_net_module.logger.name and r.getMessage().startswith("fit ")
    ]
    assert fits == [
        "fit holdout=-1 fill=none method=bayes-net",
        "fit holdout=2 fill=gap method=bayes-net",
    ]


def test_a_code_no_training_row_holds_is_no_state_and_draws_no_crash():
    train = TRAIN.loc[TRAIN["restecg"] != 1]
    bins = discretize(train, UCI_FRAME)
    assert 1 not in bins.codes["restecg"]
    assert bins.arity("restecg") == len(UCI_FRAME.codebook["restecg"]) - 1
    rows = draw(learn(bins, first="restecg"), bins, train, 500, np.random.default_rng(0), UCI_FRAME)
    assert 1 not in set(rows["restecg"])


def forced(field: str, state: int, bins: Bins) -> Network:
    """A network whose every field is drawn alone, `field` always in `state`."""
    nodes = []
    for name in bins.states.columns:
        arity = bins.arity(name)
        if name == field:
            chances = tuple(1.0 if s == state else 0.0 for s in range(arity))
        else:
            chances = tuple([1.0 / arity] * arity)
        nodes.append(Node(name, (), {(): chances}, arity))
    return Network(tuple(nodes))


def test_each_value_is_copied_from_a_donor_in_the_drawn_state():
    bins = discretize(TRAIN, UCI_FRAME)
    for field, state in (("chol", 0), ("chol", bins.arity("chol") - 1), ("cp", 2)):
        rows = draw(
            forced(field, state, bins), bins, TRAIN, 200, np.random.default_rng(2), UCI_FRAME
        )
        donors = set(TRAIN[field][bins.states[field] == state])
        assert set(rows[field]) <= donors, (field, state)
        others = set(TRAIN[field][bins.states[field] != state])
        assert set(rows[field]).isdisjoint(others - donors)


def test_the_band_edge_is_a_cut_where_no_quantile_falls_on_it():
    edge = UCI_FRAME.edges[0]
    ages = [30, 31, 32, 33, 34, edge - 2, edge + 3, 70, 71, 72, 73, 74, 75, 76, 77]
    train = pd.concat([TRAIN.iloc[[i % len(TRAIN)]] for i in range(len(ages))], ignore_index=True)
    train["age"] = ages
    bins = discretize(train, UCI_FRAME)
    quantiles = np.quantile(np.asarray(ages, dtype=float), [0.2, 0.4, 0.6, 0.8])
    assert float(edge) not in set(quantiles)
    assert float(edge) in bins.cuts["age"]
    for field in UCI_FRAME.numeric:
        if field != "age":
            assert float(edge) not in bins.cuts[field] or float(edge) in set(
                np.quantile(train[field].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
            )


def test_each_age_state_lies_in_one_band():
    bins = discretize(TRAIN, UCI_FRAME)
    bands = np.searchsorted(np.asarray(UCI_FRAME.edges), TRAIN["age"].to_numpy(), side="right")
    for state in set(bins.states["age"]):
        assert len(set(bands[bins.states["age"] == state])) == 1


def test_a_cut_at_the_largest_value_is_dropped():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert _occupied(values, [2.0, 4.0]) == (2.0, 4.0)
    assert _occupied(values, [2.0, 5.0]) == (2.0,)
    assert _occupied(values, [1.0, 2.0]) == (2.0,)


def test_a_measured_field_of_distinct_values_gets_bins_states():
    train = pd.concat([TRAIN] * 5, ignore_index=True)
    train["chol"] = np.arange(150, 150 + len(train))
    bins = discretize(train, UCI_FRAME)
    assert bins.arity("chol") == BINS
    counts = np.bincount(bins.states["chol"])
    assert counts.max() - counts.min() <= 1


def test_coded_states_are_each_codes_place_in_its_field_order():
    codes = {"cp": (4, 1, 3)}
    found = _states(pd.Series([1, 3, 4, 4, 1]), "cp", {}, codes)
    assert found.tolist() == [1, 2, 0, 0, 1]
    assert found.dtype == np.dtype(int)


def test_sparse_codes_map_without_a_table_spanning_their_range():
    codes = {"cp": (-(10**15), 7, 10**15)}
    found = _states(pd.Series([10**15, -(10**15), 7]), "cp", {}, codes)
    assert found.tolist() == [2, 0, 1]


@pytest.mark.parametrize(
    "column",
    [
        pd.Series([1.0, np.nan]),
        pd.Series([1, None], dtype=object),
        pd.Series([1, pd.NA], dtype="Int64"),
        pd.Series([1.0, 1.5]),
    ],
    ids=["nan", "none", "na", "fraction"],
)
def test_a_coded_field_with_a_missing_or_fractional_value_is_refused(column):
    with pytest.raises(ValueError, match="cp holds values that are not whole codes"):
        _states(column, "cp", {}, {"cp": (1, 2)})


def test_a_code_outside_its_fields_states_is_refused():
    with pytest.raises(KeyError, match=r"cp holds codes \[2, 9\] outside its states"):
        _states(pd.Series([1, 9, 2, 1]), "cp", {}, {"cp": (1, 3)})


def test_a_tie_falls_to_the_first_field_in_the_frames_order():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 400)
    network = learn(states_of(a=a, b=a.copy(), c=a.copy()), k=1, first="a")
    assert [node.field for node in network.nodes] == ["a", "b", "c"]
    assert network.nodes[2].parents == ("a",)


def _table_digest(network: Network) -> str:
    """The sha256 of every node's field, parents and table, each probability as its float hex."""
    text = repr(
        [
            (
                node.field,
                node.parents,
                sorted(
                    (key, tuple(float(p).hex() for p in chances))
                    for key, chances in node.table.items()
                ),
            )
            for node in network.nodes
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_learned_edges_and_tables_are_unchanged_on_the_uci_fixture():
    network = learn(discretize(TRAIN, UCI_FRAME), first="age")
    assert network.edges == [
        ("age", "chol"),
        ("age", "thalach"),
        ("chol", "thalach"),
        ("chol", "trestbps"),
        ("thalach", "trestbps"),
        ("thalach", "oldpeak"),
        ("trestbps", "oldpeak"),
        ("thalach", "cp"),
        ("oldpeak", "cp"),
        ("age", "ca"),
        ("cp", "ca"),
        ("age", "fbs"),
        ("cp", "fbs"),
        ("age", "exang"),
        ("cp", "exang"),
        ("age", "restecg"),
        ("trestbps", "restecg"),
        ("age", "slope"),
        ("restecg", "slope"),
        ("age", "thal"),
        ("restecg", "thal"),
        ("trestbps", "target"),
        ("oldpeak", "target"),
        ("chol", "sex"),
        ("trestbps", "sex"),
    ]
    assert _table_digest(network) == (
        "b5da690ca23182f3f58735931ce394f1518cc492ded591d0a282e0ff0343d316"
    )
    for node in network.nodes:
        assert all(type(state) is int for key in node.table for state in key), node.field


@pytest.mark.parametrize(
    ("first", "seed", "rows", "sha"),
    [
        ("age", 5, 200, "8804f6e8ba52d5b451d7a045538ad8a312a973d3fb47a53faf1b33e181a21b04"),
        ("age", 11, 3000, "ceeed73f3f67a010fc579e36a12937a4fc768f500d6d7cdb1637b86d59bc605e"),
        ("restecg", 7, 500, "dba45ca91f10d27887ab913ef53bca76a07a2ab66f559c19030bf9f2bac9282d"),
        ("chol", 0, 1, "7fb07f317d4365fad21e2c5cd958d06a1d2067fd6bff51c5a857637e61ae5d49"),
    ],
)
def test_draws_are_unchanged_for_a_seed(first, seed, rows, sha):
    bins = discretize(TRAIN, UCI_FRAME)
    drawn = draw(
        learn(bins, first=first), bins, TRAIN, rows, np.random.default_rng(seed), UCI_FRAME
    )
    assert hashlib.sha256(drawn.to_csv(index=False).encode()).hexdigest() == sha


def _scalar_draw(
    network: Network,
    bins: Bins,
    train: pd.DataFrame,
    rows: int,
    rng: np.random.Generator,
    frame: FrameSpec,
) -> pd.DataFrame:
    """The row-by-row draw the vector draw must equal, verbatim.

    It reads numpy's `Generator.choice` as the locked numpy implements it; a numpy upgrade that
    changes `choice` changes this oracle, and the frozen shas say whether the rows moved.
    """
    drawn: dict[str, np.ndarray] = {}
    for node in network.nodes:
        given = list(zip(*(drawn[parent] for parent in node.parents), strict=True))
        drawn[node.field] = np.array(
            [rng.choice(node.arity, p=node.chances(parents)) for parents in (given or [()] * rows)],
            dtype=int,
        )
    out = pd.DataFrame(index=range(rows))
    for field in frame.columns:
        values = train[field].to_numpy()
        donors = {
            state: values[bins.states[field].to_numpy() == state] for state in set(drawn[field])
        }
        out[field] = [rng.choice(donors[state]) for state in drawn[field]]
    return out.astype(dict(frame.dtypes))


@functools.cache
def _oracle_cases() -> dict[str, tuple]:
    """Each (name, network, bins, train, frame) the vector draw is checked on."""
    bins = discretize(TRAIN, UCI_FRAME)
    cases = {
        f"uci-{first}": (learn(bins, first=first), bins, TRAIN, UCI_FRAME)
        for first in ("age", "restecg", "chol")
    }
    for field, state in (("chol", 0), ("cp", 2)):
        cases[f"forced-{field}"] = (forced(field, state, bins), bins, TRAIN, UCI_FRAME)
    toy_bins = discretize(TOY_TRAIN, TOY)
    toy_net = learn(toy_bins, first="age")
    cases["toy"] = (toy_net, toy_bins, TOY_TRAIN, TOY)
    # an unseen parent configuration falls to the uniform prior
    sparse = Network(
        tuple(
            Node(
                node.field,
                node.parents,
                dict(list(node.table.items())[: max(1, len(node.table) // 2)]),
                node.arity,
            )
            for node in toy_net.nodes
        )
    )
    cases["toy-unseen-config"] = (sparse, toy_bins, TOY_TRAIN, TOY)
    # a field of one state
    single = TRAIN.loc[TRAIN["restecg"] == 0]
    single_bins = discretize(single, UCI_FRAME)
    assert single_bins.arity("restecg") == 1
    cases["arity-1"] = (learn(single_bins, first="age"), single_bins, single, UCI_FRAME)
    # a state whose donor pool holds one row
    thin = Bins(
        toy_bins.cuts,
        toy_bins.codes,
        toy_bins.states.assign(
            level=np.where(
                np.arange(len(toy_bins.states)) == 0, 0, toy_bins.states["level"].clip(1)
            )
        ),
    )
    assert (thin.states["level"] == 0).sum() == 1
    cases["singleton-pool"] = (learn(thin, first="age"), thin, TOY_TRAIN, TOY)
    return cases


@pytest.mark.parametrize(
    "name",
    [
        "uci-age",
        "uci-restecg",
        "uci-chol",
        "forced-chol",
        "forced-cp",
        "toy",
        "toy-unseen-config",
        "arity-1",
        "singleton-pool",
    ],
)
def test_the_vector_draw_equals_the_scalar_draw(name):
    network, bins, train, frame = _oracle_cases()[name]
    for seed in range(20):
        rows = 1 + (97 * seed) % 400
        new = draw(network, bins, train, rows, np.random.default_rng(seed), frame)
        old = _scalar_draw(network, bins, train, rows, np.random.default_rng(seed), frame)
        pd.testing.assert_frame_equal(new, old)


def test_the_vector_draw_leaves_the_generator_where_the_scalar_draw_does():
    bins = discretize(TRAIN, UCI_FRAME)
    network = learn(bins, first="age")
    new, ref = np.random.default_rng(9), np.random.default_rng(9)
    for rows in [50] * 100 + [10000] * 3:
        got = draw(network, bins, TRAIN, rows, new, UCI_FRAME)
        want = _scalar_draw(network, bins, TRAIN, rows, ref, UCI_FRAME)
        assert new.bit_generator.state == ref.bit_generator.state
        pd.testing.assert_frame_equal(got, want)


def test_a_zero_row_draw_reads_nothing():
    bins = discretize(TRAIN, UCI_FRAME)
    rng = np.random.default_rng(4)
    before = rng.bit_generator.state
    rows = draw(learn(bins, first="age"), bins, TRAIN, 0, rng, UCI_FRAME)
    assert len(rows) == 0
    assert list(rows.columns) == list(UCI_FRAME.columns)
    assert rng.bit_generator.state == before


class _Counting:
    """A generator proxy that counts `random` and `integers` calls and refuses `choice`."""

    def __init__(self, seed: int) -> None:
        self.inner = np.random.default_rng(seed)
        self.calls = 0

    def random(self, *args, **kwargs):
        self.calls += 1
        return self.inner.random(*args, **kwargs)

    def integers(self, *args, **kwargs):
        self.calls += 1
        return self.inner.integers(*args, **kwargs)

    def choice(self, *args, **kwargs):
        msg = "the draw must not call choice"
        raise AssertionError(msg)


def test_the_draw_reads_the_generator_independently_of_rows():
    bins = discretize(TRAIN, UCI_FRAME)
    network = learn(bins, first="age")
    totals = []
    for rows in (50, 500):
        counting = _Counting(1)
        draw(network, bins, TRAIN, rows, counting, UCI_FRAME)  # ty: ignore[invalid-argument-type]
        totals.append(counting.calls)
    assert totals[0] == totals[1] == len(network.nodes) + len(UCI_FRAME.columns)


class _Scripted(_Counting):
    """A generator proxy whose `random` returns one fixed value for every row."""

    def __init__(self, seed: int, value: float) -> None:
        super().__init__(seed)
        self.value = value

    def random(self, size):
        self.calls += 1
        return np.full(size, self.value)


def _one_node(chances: tuple[float, ...]) -> tuple[Network, Bins]:
    bins = states_of(level=[0, 1, 2, 0, 1, 2])
    return Network((Node("level", (), {(): chances}, 3),)), bins


def _level_frame() -> FrameSpec:
    """A stand-in carrying the two attributes `draw` reads: one `level` column of `int64`."""
    return SimpleNamespace(columns=["level"], dtypes={"level": "int64"})  # ty: ignore[invalid-return-type]


def test_a_uniform_of_zero_skips_a_zero_chance_first_state():
    network, bins = _one_node((0.0, 0.5, 0.5))
    train = pd.DataFrame({"level": [0, 1, 2, 0, 1, 2]})
    out = draw(network, bins, train, 5, _Scripted(0, 0.0), _level_frame())  # ty: ignore[invalid-argument-type]
    assert set(out["level"]) == {1}


def test_unnormalized_chances_draw_as_their_normalized_form():
    bins = states_of(level=[0, 1, 2, 0, 1, 2])
    train = pd.DataFrame({"level": [0, 1, 2, 0, 1, 2]})
    frame = _level_frame()
    raw = Network((Node("level", (), {(): (0.2, 0.6, 1.0)}, 3),))
    unit = Network((Node("level", (), {(): (0.2 / 1.8, 0.6 / 1.8, 1.0 / 1.8)}, 3),))
    for seed in range(5):
        got = draw(raw, bins, train, 300, np.random.default_rng(seed), frame)
        want = _scalar_draw(unit, bins, train, 300, np.random.default_rng(seed), frame)
        pd.testing.assert_frame_equal(got, want)


@pytest.mark.parametrize(
    "chances", [(0.0, 0.0, 0.0), (math.nan, 0.5, 0.5), (-0.5, 1.0, 0.5), (math.inf, 0.0, 0.0)]
)
def test_chances_that_cannot_be_drawn_are_refused(chances):
    network, bins = _one_node(chances)
    train = pd.DataFrame({"level": [0, 1, 2, 0, 1, 2]})
    with pytest.raises(ValueError, match="chances of level"):
        draw(network, bins, train, 5, np.random.default_rng(0), _level_frame())


def test_a_drawn_state_with_no_donor_is_refused():
    bins = states_of(level=[0, 1, 0, 1, 0, 1])
    train = pd.DataFrame({"level": [0, 1, 0, 1, 0, 1]})
    network = Network((Node("level", (), {(): (0.0, 0.0, 1.0)}, 3),))
    with pytest.raises(ValueError, match="no training donor"):
        draw(network, bins, train, 5, np.random.default_rng(0), _level_frame())


def test_mutual_information_is_computed_once_per_field_and_parents(monkeypatch):
    calls: list[tuple[str, tuple[str, ...]]] = []

    def counted(states: pd.DataFrame, field: str, parents) -> float:
        calls.append((field, tuple(parents)))
        return mutual_information(states, field, parents)

    monkeypatch.setattr(bayes_net_module, "mutual_information", counted)
    learn(discretize(TRAIN, UCI_FRAME), first="age")
    assert calls
    assert len(calls) == len(set(calls))


def test_a_classical_or_network_shortfall_holds_its_baseline(monkeypatch):
    monkeypatch.setattr(bayes_net_module, "fill", functools.partial(fill, batch=10, limit=10))
    quota: dict[Stratum, int] = {(f">={EDGE}", "female", 1): 10}
    with pytest.raises(ShortfallError) as caught:
        bayes_net(TRAIN, Request(quota=quota, seed=3))
    short = caught.value
    assert short.baseline is not None
    assert set(short.short) == {f">={EDGE}/female/1"}
    pd.testing.assert_frame_equal(short.baseline.added, short.added)
    pd.testing.assert_series_equal(short.baseline.weights, unit_weights(TRAIN))
    assert short.baseline.params["short"] == dict(short.short)
    assert short.baseline.params["seed"] == 3
