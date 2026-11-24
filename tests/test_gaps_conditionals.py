"""The conditional S-node of each mapped `is_a` pair, built on request beside the priors."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.contradictions import is_a_pairs
from bko.gaps.definitions import Definition, gaps
from bko.gaps.detect import ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.observe import (
    Observed,
    conditional_fragment,
    dataset_fragment,
    observe_records,
    record_cells,
)
from bko.gaps.shares import draw_order, label_only, nested, shares
from bko.generate.direct import _Plan
from bko.generate.presence import Presence
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import FieldMapping
from tests._contradiction_fixtures import (
    BASE_ROWS,
    BROAD,
    INNER,
    KINDS,
    NAME,
    NARROW,
    OUTER,
    ontologies,
    plant,
)
from tests._validate_fixtures import planted
from tests.test_generate_direct import frame


PAIR = (NARROW, BROAD)


def records(*shown: tuple[int, int]) -> pd.DataFrame:
    """One record per `(cp, exang)`: `cp` 1 shows `NARROW`, `exang` 1 shows `BROAD`."""
    return frame([planted(cp=cp, exang=exang) for cp, exang in shown])


def conditionals(observed_frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    """The tailed S-nodes of the frame's conditional fragment, `(tail, head)` to probability."""
    observed = observe_records(observed_frame, BASE_ROWS, name=NAME, pairs=[PAIR])
    fragment = conditional_fragment(observed)
    found = {}
    for snode in fragment.snodes:
        if snode.tail:
            ((tail, tail_state),) = snode.tail
            assert snode.head[1] == INDIVIDUAL
            assert tail_state == INDIVIDUAL
            found[tail, snode.head[0]] = snode.prob
    return found


def test_a_conditional_equals_the_hand_count():
    split = records((1, 1), (1, 1), (1, 1), (1, 0), (4, 1), (4, 0))
    observed = observe_records(split, BASE_ROWS, name=NAME, pairs=[PAIR])
    assert observed.holders[NARROW] == 4
    assert observed.both == {PAIR: 3}
    assert conditionals(split) == {PAIR: 0.75}


def test_no_conditional_where_no_record_shows_the_narrow_concept():
    assert conditionals(records((4, 1), (4, 0))) == {}


def test_a_conditional_of_probability_0_is_kept():
    assert conditionals(records((1, 0), (4, 1))) == {PAIR: 0.0}


def test_a_conditional_whose_broad_concept_no_record_shows_adds_its_inode():
    split = records((1, 0), (1, 0))
    observed = observe_records(split, BASE_ROWS, name=NAME, pairs=[PAIR])
    fragment = conditional_fragment(observed)
    assert (BROAD, INDIVIDUAL) in fragment.inodes
    assert all(snode.tail for snode in fragment.snodes if snode.head[0] == BROAD)
    assert conditionals(split) == {PAIR: 0.0}


def test_dataset_fragment_holds_no_tailed_snode():
    split = records((1, 1), (1, 0), (4, 1))
    observed = observe_records(split, BASE_ROWS, name=NAME, pairs=[PAIR])
    assert observed.both == {PAIR: 1}
    plain = observe_records(split, BASE_ROWS, name=NAME)
    fragment = dataset_fragment(observed)
    assert not any(snode.tail for snode in fragment.snodes)
    assert fragment.inodes == dataset_fragment(plain).inodes
    assert [(s.head, s.prob) for s in fragment.snodes] == [
        (s.head, s.prob) for s in dataset_fragment(plain).snodes
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_shares_and_detect_read_the_same_values_with_pairs_observed(kind):
    found = plant(kind)
    concepts = sorted({row.concept for row in found.rows})
    pairs = is_a_pairs(concepts, found.ontologies)
    assert pairs
    fragments = ontology_fragments(found.ontologies, set(concepts))
    plain = record_cells(found.frame, found.rows, name=NAME, edges=UCI_FRAME.edges)
    paired = record_cells(found.frame, found.rows, name=NAME, edges=UCI_FRAME.edges, pairs=pairs)
    assert paired.whole.both
    for definition in Definition:
        assert gaps(definition, fragments, paired) == gaps(definition, fragments, plain)
    for cell, records_of in paired.each.items():
        members = found.frame.loc[found.frame[UCI_FRAME.label].eq(cell[2])]
        group = observe_records(members, found.rows, name=f"{NAME}/group")
        group_paired = observe_records(members, found.rows, name=f"{NAME}/group", pairs=pairs)
        assert shares(records_of, group_paired) == shares(plain.each[cell], group)


def test_each_cell_counts_its_own_records_showing_both():
    split = frame(
        [
            planted(sex=1, cp=1, exang=1),
            planted(sex=1, cp=1, exang=0),
            planted(sex=0, cp=1, exang=1),
            planted(sex=0, cp=1, exang=1),
            planted(sex=0, cp=1, exang=1),
            planted(sex=0, cp=1, exang=0),
            planted(sex=0, cp=4, exang=1),
        ]
    )
    cells = record_cells(split, BASE_ROWS, name=NAME, edges=UCI_FRAME.edges, pairs=[PAIR])
    assert cells.whole.both == {PAIR: 4}
    held = {cell[1]: each.observed for cell, each in cells.each.items() if each.observed.records}
    assert {sex: observed.both[PAIR] for sex, observed in held.items()} == {
        "male": 1,
        "female": 3,
    }
    found = {}
    for sex, observed in held.items():
        (snode,) = [snode for snode in conditional_fragment(observed).snodes if snode.tail]
        found[sex] = snode.prob
    assert found == {"male": 0.5, "female": 0.75}
    assert all(
        not each.observed.both[PAIR] for each in cells.each.values() if not each.observed.records
    )


def test_conditionals_come_out_in_sorted_pair_order():
    split = frame([planted(cp=1, exang=1, restecg=2), planted(cp=1, exang=0, restecg=1)])
    inner = (INNER, OUTER)
    observed = observe_records(split, BASE_ROWS, name=NAME, pairs=[inner, PAIR])
    tailed = [
        (next(iter(snode.tail))[0], snode.head[0])
        for snode in conditional_fragment(observed).snodes
        if snode.tail
    ]
    assert tailed == sorted([inner, PAIR])
    assert tailed == [PAIR, inner]


def test_both_is_read_only():
    observed = observe_records(records((1, 1)), BASE_ROWS, name=NAME, pairs=[PAIR])
    with pytest.raises(TypeError):
        observed.both[PAIR] = 0  # ty: ignore[invalid-assignment]


def _observe_with(pair: tuple[str, str]) -> object:
    return observe_records(records((1, 1)), BASE_ROWS, name=NAME, pairs=[pair])


def _cells_with(pair: tuple[str, str]) -> object:
    return record_cells(records((1, 1)), BASE_ROWS, name=NAME, edges=UCI_FRAME.edges, pairs=[pair])


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_observe_with, id="observe_records"),
        pytest.param(_cells_with, id="record_cells"),
    ],
)
@pytest.mark.parametrize(
    ("pair", "named"),
    [
        pytest.param((NARROW, "HP:0000001"), "HP:0000001", id="unmapped"),
        pytest.param((NARROW, NARROW), NARROW, id="itself"),
    ],
)
def test_a_pair_the_rows_cannot_count_is_refused(reader, pair, named):
    with pytest.raises(GapError, match=re.escape(named)):
        reader(pair)


def test_observed_refuses_a_pair_naming_a_concept_without_a_count():
    with pytest.raises(GapError, match=r"no count for: x, y$"):
        Observed(name=NAME, records=2, holders={"a": 1}, both={("x", "y"): 0})


def test_observed_refuses_a_both_count_above_either_concepts_holders():
    with pytest.raises(GapError, match=r"counts 2 records showing both a and b, more than the 1"):
        Observed(name=NAME, records=3, holders={"a": 3, "b": 1}, both={("a", "b"): 2})
    kept = Observed(name=NAME, records=3, holders={"a": 3, "b": 1}, both={("a", "b"): 1})
    assert kept.both == {("a", "b"): 1}


def test_draw_order_puts_a_narrow_concept_before_its_broad_one():
    assert draw_order(["a", "b", "c", "d"], [("d", "b"), ("b", "a")]) == ("c", "d", "b", "a")
    assert draw_order(["b", "a"], []) == ("a", "b")


def test_draw_order_sets_a_depth_one_past_the_deepest_narrow_concept():
    pairs = [("d", "b"), ("b", "a"), ("c", "a")]
    assert draw_order(["a", "b", "c", "d"], pairs) == ("c", "d", "b", "a")


def test_draw_order_ignores_a_pair_with_an_end_outside_the_concepts():
    assert draw_order(["a", "b"], [("a", "b"), ("b", "z"), ("z", "a")]) == ("a", "b")


def test_draw_order_names_exactly_a_cycle_and_the_concepts_above_it():
    pairs = [("a", "b"), ("b", "a"), ("b", "c")]
    with pytest.raises(ValueError, match="hold a cycle or depend on one") as raised:
        draw_order(["a", "b", "c", "d"], pairs)
    names = re.fullmatch(r"the pairs over (.*) hold a cycle or depend on one", str(raised.value))
    assert names is not None
    assert names.group(1).split(", ") == ["a", "b", "c"]


def _plan_order(rows: tuple[FieldMapping, ...], releases: Mapping) -> tuple[str, ...]:
    return tuple(_Plan(rows, releases, UCI_FRAME, Presence.SHARES)._order)


@pytest.mark.parametrize(
    "rows",
    [pytest.param(BASE_ROWS, id="fixture"), pytest.param(FIELD_MAPPINGS, id="uci")],
)
def test_draw_order_equals_the_plans_order_on_nested_pairs(rows):
    concepts = [
        concept
        for concept in dict.fromkeys(row.concept for row in rows)
        if concept not in label_only(rows, frame=UCI_FRAME)
    ]
    pairs = nested(rows).items()
    assert pairs
    assert draw_order(concepts, pairs) == _plan_order(rows, ontologies())


def test_draw_order_puts_the_fixtures_inner_concept_before_its_outer_one():
    order = draw_order([OUTER, INNER], nested(BASE_ROWS).items())
    assert order.index(INNER) < order.index(OUTER)
