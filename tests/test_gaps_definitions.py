"""The three gap definitions: per-subgroup, population-level and frequency-aware cell gaps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import pytest

import bko.gaps.definitions
from bko.bkb.errors import QueryError
from bko.bkb.structure import BKB
from bko.gaps.definitions import (
    CREDIBILITY,
    SHARE_RATIO,
    Definition,
    gaps,
    posterior_below,
    under_represented,
)
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.observe import Cell, CellRecords, Cells, Observed, record_cells
from tests._gaps_fixtures import ANGINA, STENOSIS, parse_ladders


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

A = ANGINA.concept
B = STENOSIS.concept
CHEST = "HP:9000002"
M3 = "MONDO:9000003"
H21 = Statement("hpo", "HP:9000002", "HP:9000001")
H32 = Statement("hpo", "HP:9000003", "HP:9000002")
M21 = Statement("mondo", "MONDO:9000002", "MONDO:9000001")
M32 = Statement("mondo", "MONDO:9000003", "MONDO:9000002")
M42 = Statement("mondo", "MONDO:9000004", "MONDO:9000002")


@pytest.fixture
def ladders(tmp_path):
    return parse_ladders(tmp_path)


def cell(
    records: int,
    holders: Mapping[str, int],
    *,
    label: int | None = 0,
    assumed: Iterable[str] = (),
) -> CellRecords:
    return CellRecords(
        Observed(name="t", records=records, holders=holders), label, frozenset(assumed)
    )


def dataset(each: Mapping[Cell, CellRecords]) -> Cells:
    """`each`, every cell holding every concept a cell names, with their sum as the whole."""
    concepts = sorted({concept for one in each.values() for concept in one.observed.holders})
    full: dict[Cell, CellRecords] = {
        key: cell(
            one.observed.records,
            {concept: one.observed.holders.get(concept, 0) for concept in concepts},
            label=one.label,
            assumed=one.assumed,
        )
        for key, one in each.items()
    }
    whole = Observed(
        name="t",
        records=sum(one.observed.records for one in full.values()),
        holders={c: sum(one.observed.holders[c] for one in full.values()) for c in concepts},
    )
    return Cells(whole=whole, each=full)


def found(
    definition: Definition, fragments: Sequence[BKB], each: Mapping[Cell, CellRecords]
) -> list[tuple[Cell, Statement]]:
    return [(gap.cell, gap.statement) for gap in gaps(definition, fragments, dataset(each))]


@pytest.mark.parametrize(
    ("records", "below", "above"), [(1, 0.964, 0.965), (2, 0.711, 0.712), (3, 0.555, 0.556)]
)
def test_a_small_cell_showing_nothing_is_flagged_only_past_its_threshold(records, below, above):
    assert posterior_below(0, records, SHARE_RATIO * below) < CREDIBILITY
    assert posterior_below(0, records, SHARE_RATIO * above) >= CREDIBILITY


def test_the_posterior_is_the_jeffreys_beta():
    x, y = 0.5, 0.7
    assert posterior_below(0, 0, x) == pytest.approx(2 / math.pi * math.asin(math.sqrt(x)))
    assert posterior_below(0, 1, x) == pytest.approx(
        2 / math.pi * (math.asin(math.sqrt(x)) + math.sqrt(x * (1 - x)))
    )
    assert posterior_below(1, 1, 1 - y) == pytest.approx(
        1 - 2 / math.pi * (math.asin(math.sqrt(y)) + math.sqrt(y * (1 - y)))
    )


def test_an_empty_cell_is_never_flagged():
    mass = posterior_below(0, 0, SHARE_RATIO * 1.0)
    assert mass == pytest.approx(0.7048327646991335)
    assert mass < CREDIBILITY


def test_the_reference_is_the_rest_of_the_cells_label_group():
    each: dict[Cell, CellRecords] = {
        ("a", 0): cell(30, {A: 0}, label=0),
        ("b", 0): cell(30, {A: 0}, label=0),
        ("a", 1): cell(30, {A: 30}, label=1),
        ("b", 1): cell(30, {A: 0}, label=1),
    }
    flags = under_represented(dataset(each))
    assert [
        (f.cell, f.concept, f.shown, f.records, f.reference_shown, f.reference_records)
        for f in flags
    ] == [(("b", 1), A, 0, 30, 30, 30)]
    assert flags[0].mass == pytest.approx(posterior_below(0, 30, SHARE_RATIO))


def test_the_cell_is_left_out_of_its_own_reference():
    each: dict[Cell, CellRecords] = {("p",): cell(3, {A: 0}), ("q",): cell(3, {A: 3})}
    flags = under_represented(dataset(each))
    assert [(f.cell, f.reference_shown, f.reference_records) for f in flags] == [(("p",), 3, 3)]


def test_nothing_is_flagged_without_a_record_outside_the_cell_or_a_reference_share():
    each: dict[Cell, CellRecords] = {
        ("c",): cell(10, {A: 0}, label=0),
        ("d",): cell(0, {A: 0}, label=0),
        ("e",): cell(10, {A: 10}, label=1),
        ("f",): cell(10, {A: 0}, label=1),
    }
    assert [f.cell for f in under_represented(dataset(each))] == [("f",)]
    alone: dict[Cell, CellRecords] = {("only",): cell(10, {A: 0}, label=None)}
    assert under_represented(dataset(alone)) == ()


def test_per_subgroup_grounds_each_cell_against_its_own_records(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(2, {A: 1, B: 1}),
        ("y",): cell(2, {B: 2}),
        ("z",): cell(0, {}),
    }
    assert found(Definition.SUBGROUP, fragments, each) == [
        (("y",), H21),
        (("y",), H32),
        (("z",), H21),
        (("z",), H32),
        (("z",), M21),
        (("z",), M42),
    ]


def test_a_cell_holds_what_it_assumes_even_when_empty(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("y",): cell(2, {}, assumed={B}),
        ("z",): cell(0, {}, assumed={B}),
    }
    assert found(Definition.SUBGROUP, fragments, each) == [
        (key, statement) for key in (("y",), ("z",)) for statement in (H21, H32)
    ]


def test_population_level_gaps_hold_in_every_cell_whatever_it_assumes(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    held: dict[Cell, CellRecords] = {("x",): cell(3, {B: 3}), ("y",): cell(0, {}, assumed={A})}
    assert found(Definition.POPULATION, fragments, held) == [
        (key, statement) for key in held for statement in (H21, H32)
    ]
    nothing: dict[Cell, CellRecords] = {
        ("x",): cell(3, {}),
        ("y",): cell(0, {}, assumed={B}),
    }
    assert found(Definition.POPULATION, fragments, nothing) == [
        (key, statement) for key in nothing for statement in (H21, H32, M21, M42)
    ]


def test_a_shown_under_represented_concept_yields_the_statements_it_alone_grounds(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(27, {A: 1, B: 27}),
        ("y",): cell(20, {A: 20, B: 20}),
    }
    assert found(Definition.FREQUENCY, fragments, each) == [(("x",), H21), (("x",), H32)]


def test_a_statement_another_held_concept_grounds_is_not_a_frequency_aware_gap(ladders):
    fragments = ontology_fragments(ladders, {A, CHEST})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(20, {A: 0, CHEST: 20}),
        ("y",): cell(20, {A: 20, CHEST: 20}),
    }
    assert found(Definition.FREQUENCY, fragments, each) == [(("x",), H32)]


def test_the_frequency_aware_base_holds_what_the_cell_assumes(ladders):
    fragments = ontology_fragments(ladders, {M3, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(30, {M3: 0, B: 0}, assumed={B}),
        ("y",): cell(30, {M3: 30, B: 0}),
    }
    assert found(Definition.FREQUENCY, fragments, each) == [(("x",), M32)]


def test_a_cells_under_represented_concepts_are_left_out_together(ladders):
    fragments = ontology_fragments(ladders, {M3, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(30, {M3: 0, B: 0}),
        ("y",): cell(30, {M3: 30, B: 30}),
    }
    assert found(Definition.FREQUENCY, fragments, each) == [
        (("x",), M21),
        (("x",), M32),
        (("x",), M42),
    ]


def test_a_training_frame_gives_each_definitions_gaps(ladders):
    rows = (
        [(63, 0, 0, 1)]
        + [(63, 0, 0, 4)] * 19
        + [(63, 1, 0, 1)] * 20
        + [(63, 0, 1, 1)] * 5
        + [(63, 1, 1, 4)] * 5
    )
    frame = pd.DataFrame(rows, columns=["age", "sex", "target", "cp"], dtype="int64")
    cells = record_cells(frame, [ANGINA, STENOSIS], name="uci")
    fragments = ontology_fragments(ladders, {A, B})
    every = (H21, H32, M21, M42)

    def listed(definition: Definition) -> list[tuple[Cell, Statement]]:
        return [(gap.cell, gap.statement) for gap in gaps(definition, fragments, cells)]

    assert listed(Definition.SUBGROUP) == [
        *((("<55", "female", 0), s) for s in (H21, H32)),
        *((("<55", "female", 1), s) for s in every),
        *((("<55", "male", 0), s) for s in (H21, H32)),
        *((("<55", "male", 1), s) for s in every),
        *(((">=55", "male", 1), s) for s in (H21, H32)),
    ]
    assert listed(Definition.POPULATION) == []
    assert listed(Definition.FREQUENCY) == [
        *(((">=55", "female", 0), s) for s in (H21, H32)),
        *(((">=55", "male", 1), s) for s in (H21, H32)),
    ]


def test_an_empty_cell_is_not_flagged_against_a_reference_that_shows_the_concept():
    each: dict[Cell, CellRecords] = {("e",): cell(0, {A: 0}), ("r",): cell(30, {A: 30})}
    assert under_represented(dataset(each)) == ()


def test_a_mass_exactly_at_the_credibility_level_is_flagged(monkeypatch):
    each: dict[Cell, CellRecords] = {("p",): cell(30, {A: 0}), ("q",): cell(30, {A: 30})}
    mass = posterior_below(0, 30, SHARE_RATIO)
    monkeypatch.setattr(bko.gaps.definitions, "CREDIBILITY", mass)
    assert [f.cell for f in under_represented(dataset(each))] == [("p",)]


def test_a_cell_assuming_a_concept_no_fragment_holds_is_refused(ladders):
    fragments = ontology_fragments(ladders, {A})
    each: dict[Cell, CellRecords] = {("x",): cell(2, {A: 1}, assumed={B})}
    with pytest.raises(QueryError, match="MONDO:9000004"):
        gaps(Definition.SUBGROUP, fragments, dataset(each))
