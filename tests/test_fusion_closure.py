"""`bko.fusion.closure`: each definition's pairs before and after the join, and their share."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import starmap
from pathlib import Path

import pytest

from bko.bkb.structure import BKB
from bko.datasets.uci import UCI_FRAME
from bko.fusion.closure import Closure, closure
from bko.fusion.join import join
from bko.gaps.definitions import Definition, StatementGap
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.observe import Cell, record_cells
from bko.validate.checks import validate
from tests._fusion_fixtures import CELL0, CELL1, UCI_X, UCI_Y, training_split, young
from tests._gaps_fixtures import ANGINA, STENOSIS, parse_ladders


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

GROUNDING = (Definition.SUBGROUP, Definition.POPULATION)
S1 = Statement("hpo", "HP:9000002", "HP:9000001")
S2 = Statement("hpo", "HP:9000003", "HP:9000002")
S3 = Statement("mondo", "MONDO:9000002", "MONDO:9000001")
X: Cell = ("x", 0)
Y: Cell = ("y", 0)


def pairs(*items: tuple[Cell, Statement]) -> tuple[StatementGap, ...]:
    return tuple(starmap(StatementGap, items))


def fragments(directory: Path) -> tuple[BKB, ...]:
    return ontology_fragments(parse_ladders(directory), {ANGINA.concept, STENOSIS.concept})


def read(
    tmp_path: Path,
    data: bytes,
    request: Mapping[Cell, int],
    items: Mapping[Cell, Sequence[object]],
) -> tuple[Closure, ...]:
    """Validate `items`, join them to `data`'s training split, and read each definition."""
    train = training_split(tmp_path / "data", data)
    accepted = {cell: validate(batch, frame=UCI_FRAME).accepted for cell, batch in items.items()}
    joined = join(train, accepted, request, edges=UCI_FRAME.edges)
    rows = (ANGINA, STENOSIS)
    before = record_cells(train, rows, name="t", edges=UCI_FRAME.edges)
    after = record_cells(joined.frame, rows, name="t-joined", edges=UCI_FRAME.edges)
    return closure(fragments(tmp_path), before, after)


BOTH: dict[Cell, int] = {CELL0: 20, CELL1: 20}
SCENARIOS: dict[str, tuple[bytes, dict[Cell, int], dict[Cell, list[object]]]] = {
    "A": (UCI_X, BOTH, {CELL1: [young(1, cp=1)] * 20, CELL0: [young(0, cp=4)] * 20}),
    "B": (UCI_X, BOTH, {CELL1: [young(1, cp=1)] * 20, CELL0: [young(0, cp=1)] * 20}),
    "C": (UCI_X, BOTH, {CELL1: [young(1, cp=4)] * 20}),
    "D": (UCI_X, BOTH, {}),
    "Y": (UCI_Y, {CELL1: 20}, {CELL1: [young(1, cp=1)] * 20}),
}
MEASURED = {
    "A": ((6, 2, 4, 2, 0), (0, 0, 0, 0, 0), (2, 2, 0, 2, 0)),
    "B": ((6, 0, 6, 0, 0), (0, 0, 0, 0, 0), (2, 0, 2, 0, 0)),
    "C": ((6, 4, 2, 4, 0), (0, 0, 0, 0, 0), (2, 4, 0, 2, 2)),
    "D": ((6, 6, 0, 6, 0), (0, 0, 0, 0, 0), (2, 2, 0, 2, 0)),
    "Y": ((18, 14, 4, 14, 0), (16, 0, 16, 0, 0), (0, 6, 0, 0, 6)),
}


def counts(reading: Closure) -> tuple[int, int, int, int, int]:
    return (
        len(reading.before),
        len(reading.after),
        len(reading.closed),
        len(reading.still_open),
        len(reading.opened),
    )


def test_closed_is_before_minus_after_in_before_order():
    reading = Closure(
        Definition.SUBGROUP, pairs((X, S2), (X, S1), (Y, S1)), pairs((Y, S1), (X, S3))
    )
    assert reading.closed == pairs((X, S2), (X, S1))
    assert reading.still_open == pairs((Y, S1))


def test_new_is_after_minus_before():
    reading = Closure(Definition.SUBGROUP, pairs((X, S1)), pairs((Y, S2), (X, S1), (X, S3)))
    assert reading.opened == pairs((Y, S2), (X, S3))


def test_the_share_counts_pairs_closed_not_the_net_change():
    before = pairs((X, S1), (X, S2), (X, S3), (Y, S1), (Y, S2), (Y, S3))
    after = pairs((X, S1), (X, S2), (X, S3), (("z", 0), S1), (("z", 0), S2), (("z", 0), S3))
    reading = Closure(Definition.SUBGROUP, before, after)
    assert len(reading.closed) == 3
    assert len(reading.opened) == 3
    assert reading.share == pytest.approx(0.5)


def test_the_share_is_undefined_when_nothing_was_listed_before():
    assert Closure(Definition.POPULATION, (), pairs((X, S1))).share is None


def test_a_pair_is_a_cell_and_a_statement():
    reading = Closure(Definition.SUBGROUP, pairs((X, S1), (Y, S1)), pairs((Y, S1)))
    assert reading.closed == pairs((X, S1))
    assert reading.share == pytest.approx(0.5)


def test_listed_follows_the_grid_and_sorts_each_cells_statements():
    a: Cell = ("a", 0)
    b: Cell = ("b", 0)
    reading = Closure(Definition.SUBGROUP, pairs((a, S3), (b, S2)), pairs((b, S1), (a, S1)))
    assert reading.listed([b, a]) == pairs((b, S1), (b, S2), (a, S1), (a, S3))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_scenario_reads_as_measured(tmp_path, name):
    readings = read(tmp_path, *SCENARIOS[name])
    assert [reading.definition for reading in readings] == list(Definition)
    assert tuple(counts(reading) for reading in readings) == MEASURED[name]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_joining_records_never_opens_a_grounding_pair(tmp_path, name):
    readings = {reading.definition: reading for reading in read(tmp_path, *SCENARIOS[name])}
    for definition in GROUNDING:
        assert readings[definition].opened == ()


def test_frequency_aware_can_open_pairs(tmp_path):
    readings = {reading.definition: reading for reading in read(tmp_path, *SCENARIOS["C"])}
    opened = readings[Definition.FREQUENCY].opened
    assert opened
    assert {gap.cell for gap in opened} == {CELL1}


def test_one_cells_records_close_a_population_level_pair_in_every_cell(tmp_path):
    readings = {reading.definition: reading for reading in read(tmp_path, *SCENARIOS["Y"])}
    closed = readings[Definition.POPULATION].closed
    assert len({gap.cell for gap in closed}) == 8


def test_rejected_candidates_close_nothing(tmp_path):
    items = [young(1, cp=1, chol=601)] * 20
    assert validate(items, frame=UCI_FRAME).passed == 0
    readings = read(tmp_path, UCI_X, BOTH, {CELL1: items})
    assert tuple(counts(reading) for reading in readings) == MEASURED["D"]
