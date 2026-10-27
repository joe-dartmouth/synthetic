"""The three gap definitions: per-subgroup, population-level and frequency-aware cell gaps."""

from __future__ import annotations

import ast
import inspect
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

import bko.gaps.definitions
from bko.bkb.errors import QueryError
from bko.bkb.structure import BKB
from bko.gaps.definitions import (
    CREDIBILITY,
    SHARE_RATIO,
    Definition,
    completeness_request,
    gaps,
    posterior_below,
    under_represented,
)
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.errors import GapError
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
    definition: Definition,
    fragments: Sequence[BKB],
    each: Mapping[Cell, CellRecords],
    *,
    grounding: int = 1,
    credibility: float = CREDIBILITY,
) -> list[tuple[Cell, Statement]]:
    return [
        (gap.cell, gap.statement)
        for gap in gaps(
            definition, fragments, dataset(each), grounding=grounding, credibility=credibility
        )
    ]


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


@pytest.mark.parametrize(
    "settings", [{}, {"grounding": 1, "credibility": 0.95}], ids=["defaults", "explicit"]
)
def test_a_training_frame_gives_each_definitions_gaps(ladders, settings):
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
        return [(gap.cell, gap.statement) for gap in gaps(definition, fragments, cells, **settings)]

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


def test_a_mass_exactly_at_the_credibility_level_is_flagged():
    each: dict[Cell, CellRecords] = {("p",): cell(2, {A: 0}), ("q",): cell(20, {A: 12})}
    mass = posterior_below(0, 2, SHARE_RATIO * 12 / 20)
    assert [f.cell for f in under_represented(dataset(each), credibility=mass)] == [("p",)]
    assert under_represented(dataset(each), credibility=math.nextafter(mass, 1)) == ()


def test_the_defaults_are_the_rules_e2_ran():
    assert CREDIBILITY == 0.95  # ruff: ignore[float-equality-comparison]
    assert inspect.signature(gaps).parameters["grounding"].default == 1


def _passes_a_setting(node: ast.Call) -> bool:
    named = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    return any(
        keyword.arg in {"grounding", "credibility"}
        or (keyword.arg is None and named in {"gaps", "under_represented"})
        for keyword in node.keywords
    )


def test_outside_the_definitions_only_the_sweep_passes_a_grounding_count_or_a_level():
    package = Path(bko.gaps.definitions.__file__).parent.parent
    passing = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and _passes_a_setting(node)
    }
    assert passing == {"gaps/definitions.py", "gaps/sweep.py"}


def test_the_guard_sees_a_splat_into_gaps():
    splat, other = (ast.parse(code).body[0] for code in ("gaps(d, f, c, **s)", "dict(**s)"))
    assert isinstance(splat, ast.Expr)
    assert isinstance(other, ast.Expr)
    assert isinstance(splat.value, ast.Call)
    assert isinstance(other.value, ast.Call)
    assert _passes_a_setting(splat.value)
    assert not _passes_a_setting(other.value)


def test_a_lower_level_flags_a_mass_a_higher_one_does_not(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(2, {A: 0, B: 2}),
        ("y",): cell(20, {A: 12, B: 20}),
    }
    mass = posterior_below(0, 2, SHARE_RATIO * 12 / 20)
    assert 0.90 <= mass < 0.95
    assert found(Definition.FREQUENCY, fragments, each) == []
    assert found(Definition.FREQUENCY, fragments, each, credibility=0.90) == [
        (("x",), H21),
        (("x",), H32),
    ]


def test_a_grounding_count_of_two_drops_a_concept_one_record_shows(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(3, {A: 1, B: 2}),
        ("y",): cell(2, {B: 2}),
        ("z",): cell(0, {}, assumed={B}),
    }
    assert found(Definition.SUBGROUP, fragments, each) == [
        (key, statement) for key in (("y",), ("z",)) for statement in (H21, H32)
    ]
    assert found(Definition.SUBGROUP, fragments, each, grounding=2) == [
        (key, statement) for key in each for statement in (H21, H32)
    ]
    assert found(Definition.POPULATION, fragments, each) == []
    assert found(Definition.POPULATION, fragments, each, grounding=2) == [
        (key, statement) for key in each for statement in (H21, H32)
    ]


def test_an_assumed_concept_is_held_at_every_grounding_count(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {("z",): cell(0, {}, assumed={B})}
    assert found(Definition.SUBGROUP, fragments, each, grounding=5) == [
        (("z",), H21),
        (("z",), H32),
    ]


def test_the_frequency_aware_definition_ignores_the_grounding_count(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(27, {A: 1, B: 27}),
        ("y",): cell(20, {A: 20, B: 20}),
    }
    expected = [(("x",), H21), (("x",), H32)]
    for grounding in (1, 2, 5):
        assert found(Definition.FREQUENCY, fragments, each, grounding=grounding) == expected


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"grounding": 0}, "grounding count must be at least 1, not 0"),
        ({"credibility": 0.0}, "level must lie strictly between 0 and 1, not 0.0"),
        ({"credibility": 1.0}, "level must lie strictly between 0 and 1, not 1.0"),
    ],
)
def test_a_setting_out_of_range_is_refused(ladders, settings, message):
    fragments = ontology_fragments(ladders, {A, B})
    cells = dataset({("x",): cell(2, {A: 1, B: 1})})
    for definition in Definition:
        with pytest.raises(ValueError, match=re.escape(message)):
            gaps(definition, fragments, cells, **settings)


def test_under_represented_refuses_a_level_out_of_range():
    cells = dataset({("p",): cell(30, {A: 0}), ("q",): cell(30, {A: 30})})
    with pytest.raises(ValueError, match=r"level must lie strictly between 0 and 1, not 1\.5"):
        under_represented(cells, credibility=1.5)


def test_a_cell_assuming_a_concept_no_fragment_holds_is_refused(ladders):
    fragments = ontology_fragments(ladders, {A})
    each: dict[Cell, CellRecords] = {("x",): cell(2, {A: 1}, assumed={B})}
    with pytest.raises(QueryError, match="MONDO:9000004"):
        gaps(Definition.SUBGROUP, fragments, dataset(each))


def test_the_completeness_request_asks_twenty_of_each_cell_holding_a_per_subgroup_gap_in_grid_order(
    ladders,
):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("z",): cell(0, {}),
        ("x",): cell(2, {A: 1, B: 1}),
        ("y",): cell(2, {B: 2}),
    }
    result = completeness_request(fragments, dataset(each))
    assert result == {("z",): 20, ("y",): 20}
    assert list(result) == [("z",), ("y",)]


def test_a_cell_with_many_records_and_one_gap_is_still_asked_twenty(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {
        ("x",): cell(2, {A: 1, B: 1}),
        ("y",): cell(500, {B: 500}),
    }
    assert completeness_request(fragments, dataset(each)) == {("y",): 20}


def test_no_per_subgroup_gap_refuses_the_request(ladders):
    fragments = ontology_fragments(ladders, {A, B})
    each: dict[Cell, CellRecords] = {("x",): cell(2, {A: 1, B: 1})}
    with pytest.raises(GapError, match="holds no per-subgroup gap"):
        completeness_request(fragments, dataset(each))
