"""The conditional S-node of each mapped `is_a` pair, built on request beside the priors."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.contradictions import is_a_pairs
from bko.gaps.definitions import Definition, gaps
from bko.gaps.detect import ontology_fragments
from bko.gaps.observe import conditional_fragment, dataset_fragment, observe_records, record_cells
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


def test_draw_order_puts_a_narrow_concept_before_its_broad_one_and_refuses_a_cycle():
    assert draw_order(["a", "b", "c", "d"], [("d", "b"), ("b", "a")]) == ("c", "d", "b", "a")
    assert draw_order(["b", "a"], []) == ("a", "b")
    with pytest.raises(ValueError, match="a, b"):
        draw_order(["a", "b", "c"], [("a", "b"), ("b", "a")])


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
    if rows is BASE_ROWS:
        assert draw_order(concepts, pairs).index(INNER) < draw_order(concepts, pairs).index(OUTER)
