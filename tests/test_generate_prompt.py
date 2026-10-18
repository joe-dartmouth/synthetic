"""The prompt: the system text every level shares, and the user text of the first two levels."""

from __future__ import annotations

import pytest

from bko.datasets.cells import LABEL, LABEL_MEANINGS
from bko.datasets.uci import CODEBOOK, DESCRIPTIONS, FRAME_COLUMNS
from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import Level
from bko.generate.prompt import COHERENCE, SYSTEM, UNITS, render
from bko.ontology.mapping import RANGES
from tests._generate_fixtures import GUIDANCE


OLDER = (">=55", "male", 0)
"""An older cell, whose age band has no upper edge."""


def _line(name: str) -> str:
    """The system prompt's line for column `name`."""
    lines = [line for line in SYSTEM.splitlines() if line.startswith(f"- {name}: ")]
    assert len(lines) == 1, name
    return lines[0]


def test_the_system_prompt_names_every_column_with_its_description_codes_unit_and_plausible_range():
    for name in FRAME_COLUMNS:
        assert DESCRIPTIONS[name] in _line(name), name
    for name, codes in CODEBOOK.items():
        for code, meaning in codes.items():
            assert f"; {code} = {meaning}" in _line(name), (name, code)
    for code, meaning in LABEL_MEANINGS.items():
        assert f"; {code} = {meaning}" in _line(LABEL)
    for bound in RANGES:
        assert f"plausible from {bound.low:g} to {bound.high:g}" in _line(bound.field)
        assert UNITS[bound.unit] in _line(bound.field)
    assert _line("age").endswith("age in years; plausible from 18 to 120")
    assert "3 = normal; 6 = fixed defect; 7 = reversable defect" in _line("thal")
    assert all(f"; {code} = " in _line("cp") for code in range(1, 5))
    assert _line("chol").endswith("; mg/dL, plausible from 32 to 600")


def test_the_system_prompt_asks_for_new_patients():
    assert "Invent new patients" in SYSTEM
    assert "do not reproduce records from any published dataset" in SYSTEM


@pytest.mark.parametrize(
    ("cell", "sex", "label", "sentence"),
    [
        (GUIDANCE.cell, 0, 1, "sex 0 (female), target 1 (> 50% diameter narrowing), age 18 to 54"),
        (OLDER, 1, 0, "sex 1 (male), target 0 (< 50% diameter narrowing), age 55 to 120"),
    ],
)
def test_the_cell_level_states_sex_label_age_band_record_count_and_row_count(
    cell, sex, label, sentence
):
    guidance = Guidance(cell=cell, sex=sex, label=label, records=4, interval=intervals(cell))
    system, user = render(guidance, 3, Level.CELL)
    assert system == SYSTEM
    assert "Write exactly 3 records" in user
    assert sentence in user
    assert "The training data holds 4 records of this group." in user
    low, high = guidance.interval["age"]
    assert f"Keep each numeric field within: age {low:g} to {high:g} years;" in user
    assert "oldpeak -10 to 10 mm" in user
    assert COHERENCE not in user


def test_the_coherent_level_adds_only_the_coherence_sentence():
    _, cell = render(GUIDANCE, 2, Level.CELL)
    _, coherent = render(GUIDANCE, 2, Level.COHERENT)
    assert coherent.startswith(cell)
    assert coherent.removeprefix(cell).strip() == COHERENCE


def test_intervals_clip_age_to_the_band_within_the_plausible_range():
    young = intervals(GUIDANCE.cell)
    assert young["age"] == (18, 54)
    assert {name: bounds for name, bounds in young.items() if name != "age"} == {
        bound.field: (bound.low, bound.high) for bound in RANGES if bound.field != "age"
    }
    assert intervals(OLDER)["age"] == (55, 120)
    assert intervals(("40-59", "male", 1), edges=(40, 60))["age"] == (40, 59)
    assert intervals(("<40", "male", 1), edges=(40, 60))["age"] == (18, 39)
    with pytest.raises(ValueError, match="60-69"):
        intervals(("60-69", "male", 1))


def test_rendering_is_deterministic():
    assert render(GUIDANCE, 2, Level.SUMMARY) == render(GUIDANCE, 2, Level.SUMMARY)
