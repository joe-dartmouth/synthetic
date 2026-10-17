"""The complete-evidence log joint: pybkb's results matched, an exact reference, and refusals."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

import pytest

from bko.bkb.errors import QueryError
from bko.bkb.fusion import Normalization, fuse
from bko.bkb.reason import log_joint_complete
from bko.bkb.serialize import render
from bko.bkb.structure import BKB, State
from tests._bkb_fixtures import Query, build, exported, exported_queries, h1_fragments
from tests._bkb_reference import reference_mass
from tests._corpus import REPO_ROOT


pytestmark = pytest.mark.filterwarnings("error")

QUERIES = exported_queries()
IDS = [f"{query.bkb}-{index}" for index, query in enumerate(QUERIES)]


def close(value: float) -> Any:
    return pytest.approx(value, rel=1e-12, abs=1e-12)


def assert_matches(got: Mapping[State, float | None], expected: Mapping[str, float | None]):
    assert set(got) == set(expected)
    for state, value in expected.items():
        if value is None:
            assert got[state] is None
        elif value == -math.inf:
            assert got[state] == -math.inf
        else:
            assert got[state] == close(value)


@pytest.mark.parametrize("query", QUERIES, ids=IDS)
def test_matches_pybkbs_heuristic_update(query: Query):
    got = log_joint_complete(exported(query.bkb), query.target, query.evidence)
    assert_matches(got, query.result)


def test_the_golden_queries_cover_unsupported_zero_and_supported_states():
    values = [value for query in QUERIES for value in query.result.values()]
    assert any(value is None for value in values)
    assert any(value == -math.inf for value in values)
    assert any(value is not None and math.isfinite(value) for value in values)
    # exp() underflows near -745, where pybkb's log(exp(a) + exp(b)) loses terms; no finite
    # golden is near that, so pybkb's sum and the port's stable one should agree.
    assert all(value is None or value == -math.inf or value > -700 for value in values)


def first_two(bkb: str) -> list[int]:
    return [i for i, q in enumerate(QUERIES) if q.bkb == bkb][:2]


REFERENCED = [i for i, q in enumerate(QUERIES) if q.bkb == "aquatic_eco"] + [
    i
    for bkb in ("pirate_fused_r1", "pirate_fused_mixed", "pirate_fused_mixed_collapsed")
    for i in first_two(bkb)
]


@pytest.mark.parametrize("index", REFERENCED, ids=[IDS[i] for i in REFERENCED])
def test_agrees_with_the_exact_reference(index):
    query = QUERIES[index]
    bkb = exported(query.bkb)
    exact = reference_mass(json.loads(render(bkb)), query.target, dict(query.evidence))
    got = log_joint_complete(bkb, query.target, query.evidence)
    assert set(got) == set(exact)
    for state, mass in exact.items():
        if mass is None:
            assert got[state] is None
        elif mass == 0:
            assert got[state] == -math.inf
        else:
            assert got[state] == close(math.log(mass.numerator) - math.log(mass.denominator))


def h1() -> BKB:
    return fuse(list(h1_fragments()), [3, 1], normalization=Normalization.COMPONENT)


@pytest.mark.parametrize(
    "normalization", [Normalization.COMPONENT, Normalization.COMPONENT_COLLAPSED]
)
def test_h1_matches_its_hand_worked_values(normalization):
    # Y=1: .5 x 3/4 + .75 x 1/4 = 9/16. X=0|Y=1: .25 x 3/4 + .5 x 1/4 = 5/16, so 45/256;
    # X=1|Y=1: .75 x 3/4 + .5 x 1/4 = 11/16, so 99/256.
    fused = fuse(list(h1_fragments()), [3, 1], normalization=normalization)
    assert log_joint_complete(fused, "X", {"Y": "1"}) == {
        "0": close(math.log(45 / 256)),
        "1": close(math.log(99 / 256)),
    }


def test_the_reference_reproduces_h1_exactly():
    exact = reference_mass(json.loads(render(h1())), "X", {"Y": "1"})
    assert exact == {"0": Fraction(45, 256), "1": Fraction(99, 256)}


def test_a_zero_probability_gives_minus_infinity_without_a_warning():
    bkb = build(
        "zero", [("Y", "0", 1.0, []), ("X", "0", 0.0, [("Y", "0")]), ("X", "1", 1.0, [("Y", "0")])]
    )
    assert log_joint_complete(bkb, "X", {"Y": "0"}) == {"0": -math.inf, "1": 0.0}


def test_a_state_no_snode_supports_is_none():
    bkb = build("unsupported", [("Y", "0", 1.0, []), ("X", "0", 0.5, [("Y", "0")])])
    bkb.add_inode("X", "1")
    assert log_joint_complete(bkb, "X", {"Y": "0"}) == {"0": close(math.log(0.5)), "1": None}


def test_tiny_probabilities_stay_finite_where_pybkbs_sum_underflows():
    # Each I-node sums two sources of 1e-100 x 1/2; pybkb's log(exp(a) + exp(b)) gives -inf.
    rows = [(name, "1", 1e-100, []) for name in "ABCD"]
    fused = fuse([build("a", rows), build("b", rows)], [1, 1])
    got = log_joint_complete(fused, "A", {"B": "1", "C": "1", "D": "1"})
    assert got == {"1": close(4 * math.log(1e-100))}


def cycle() -> BKB:
    return build("cycle", [("A", "1", 0.5, [("B", "1")]), ("B", "1", 0.5, [("A", "1")])])


def test_a_cyclic_choice_is_summed_as_pybkb_sums_it():
    assert log_joint_complete(cycle(), "A", {"B": "1"}) == {"1": close(math.log(0.25))}
    assert reference_mass(json.loads(render(cycle())), "A", {"B": "1"}) == {"1": Fraction(1, 4)}


def test_an_answer_that_depends_on_a_missing_component_is_refused():
    bkb = build(
        "missing",
        [("Y", "0", 1.0, []), ("Z", "0", 1.0, []), ("X", "0", 0.5, [("Y", "0"), ("Z", "0")])],
    )
    with pytest.raises(QueryError, match="leaves out"):
        log_joint_complete(bkb, "X", {"Y": "0"})


def test_a_contradiction_outranks_a_missing_component():
    bkb = build(
        "contradiction",
        [("Y", "0", 1.0, []), ("Z", "0", 1.0, []), ("X", "0", 0.5, [("Y", "1"), ("Z", "0")])],
    )
    assert log_joint_complete(bkb, "X", {"Y": "0"}) == {"0": None}


def test_a_contradiction_anywhere_outranks_a_missing_component_anywhere():
    bkb = build(
        "order",
        [
            ("W", "0", 1.0, [("Z", "0")]),
            ("Y", "0", 1.0, [("V", "1")]),
            ("V", "0", 1.0, []),
            ("V", "1", 1.0, []),
            ("X", "0", 0.5, []),
            ("X", "1", 0.5, []),
            ("Z", "0", 1.0, []),
        ],
    )
    for evidence in ({"W": "0", "Y": "0", "V": "0"}, {"V": "0", "Y": "0", "W": "0"}):
        assert log_joint_complete(bkb, "X", evidence) == {"0": None, "1": None}


def test_an_evidence_inode_without_snodes_makes_every_state_none():
    bkb = build("bare", [("X", "0", 1.0, [])])
    bkb.add_inode("Y", "0")
    assert log_joint_complete(bkb, "X", {"Y": "0"}) == {"0": None}


def test_a_tail_with_two_sources_multiplies_both_priors():
    bkb = build(
        "two sources",
        [
            ("A", "1", 0.5, [("__Source__[A]", "s"), ("__Source__[B]", "t")]),
            ("__Source__[A]", "s", 0.25, []),
            ("__Source__[B]", "t", 0.75, []),
        ],
    )
    assert log_joint_complete(bkb, "A", {}) == {"1": close(math.log(0.5 * 0.25 * 0.75))}


def mixed_support() -> BKB:
    return build("mixed", [("X", "0", 0.5, []), ("X", "0", 0.5, [("Z", "0")]), ("Z", "0", 1.0, [])])


NOT_A_STATE: Any = ["1"]


def two_priors() -> BKB:
    return build(
        "two priors",
        [
            ("Y", "0", 1.0, [("__Source__[Y]", "s")]),
            ("__Source__[Y]", "s", 0.5, []),
            ("__Source__[Y]", "s", 0.5, []),
            ("X", "0", 1.0, [("Y", "0")]),
        ],
    )


def no_prior() -> BKB:
    return build(
        "no prior", [("Y", "0", 1.0, [("__Source__[Y]", "s")]), ("X", "0", 1.0, [("Y", "0")])]
    )


REFUSED: dict[str, Callable[[], object]] = {
    "an unknown target": lambda: log_joint_complete(h1(), "W", {"Y": "1"}),
    "a source target": lambda: log_joint_complete(h1(), "__Source__[X]", {"Y": "1"}),
    "the target in the evidence": lambda: log_joint_complete(h1(), "X", {"X": "0", "Y": "1"}),
    "an unknown evidence state": lambda: log_joint_complete(h1(), "X", {"Y": "2"}),
    "source evidence": lambda: log_joint_complete(h1(), "X", {"Y": "1", "__Source__[Y]": "a"}),
    "a source with two priors": lambda: log_joint_complete(two_priors(), "X", {"Y": "0"}),
    "a source with no prior": lambda: log_joint_complete(no_prior(), "X", {"Y": "0"}),
    "an agreeing and an undetermined S-node": lambda: log_joint_complete(mixed_support(), "X", {}),
    "a non-string evidence state": lambda: log_joint_complete(h1(), "X", {"Y": NOT_A_STATE}),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=list(REFUSED))
def test_log_joint_complete_refuses(make):
    with pytest.raises(QueryError):
        make()


def test_a_query_leaves_the_bkb_unchanged():
    bkb = h1()
    before = render(bkb)
    log_joint_complete(bkb, "X", {"Y": "1"})
    assert render(bkb) == before


def test_the_reference_imports_nothing_from_bko():
    tree = ast.parse((REPO_ROOT / "tests" / "_bkb_reference.py").read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert names
    assert not any(name.split(".")[0] in {"bko", "tests"} for name in names)
