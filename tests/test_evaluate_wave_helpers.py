"""`bko.evaluate.waves`: a round's next wave and the lines a prescribing arm gives back."""

from __future__ import annotations

from bko.evaluate.waves import lines_requeued, next_wave
from bko.gaps.observe import Cell
from bko.generate.guidance import Guidance


def guidance(cell: Cell, sex: int, label: int) -> Guidance:
    return Guidance(cell, sex, label, records=0, interval={})


def test_a_wave_stops_at_the_first_repeated_sex_and_label_chain():
    cells: list[Cell] = [
        ("<55", "female", 1),
        (">=55", "male", 1),
        (">=55", "female", 1),
        ("<55", "male", 0),
    ]
    chains = {cells[0]: (0, 1), cells[1]: (1, 1), cells[2]: (0, 1), cells[3]: (1, 0)}
    found = {cell: guidance(cell, *chains[cell]) for cell in cells}
    assert next_wave(cells, found) == cells[:2]
    assert next_wave(cells[2:], found) == cells[2:]


def test_an_arm_that_prescribes_nothing_requeues_no_line():
    assert lines_requeued(None) == 0
