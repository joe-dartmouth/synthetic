"""The prompt: the system text every level shares, and the user text of each level built so far."""

from __future__ import annotations

import dataclasses
import json

import pytest

from bko.datasets.cells import LABEL, LABEL_MEANINGS
from bko.datasets.split import ranges
from bko.datasets.uci import CODEBOOK, DESCRIPTIONS, FRAME_COLUMNS, UCI_FRAME
from bko.evaluate.rehearse import PROMPT, RECORD_LINE
from bko.gaps.detect import Statement
from bko.generate.guidance import (
    Basis,
    Finding,
    Guidance,
    Profile,
    Relationship,
    Share,
    Spread,
    intervals,
)
from bko.generate.messages import LEVELS, Level
from bko.generate.prompt import COHERENCE, SYSTEM, UNITS, render, render_shown, system
from bko.ontology.mapping import FIELD_MAPPINGS, RANGES
from bko.ontology.schema import FieldMapping
from tests._frame_fixtures import TOY
from tests._generate_fixtures import GUIDANCE, TRAIN_A


OLDER = (">=55", "male", 0)
"""An older cell, whose age band has no upper edge."""


def _line(name: str) -> str:
    """The system prompt's line for column `name`."""
    lines = [line for line in SYSTEM.splitlines() if line.startswith(f"- {name}: ")]
    assert len(lines) == 1, name
    return lines[0]


def test_the_system_prompt_names_every_column_with_its_description_codes_and_unit_and_no_range():
    for name in FRAME_COLUMNS:
        assert DESCRIPTIONS[name] in _line(name), name
    for name, codes in CODEBOOK.items():
        for code, meaning in codes.items():
            assert f"; {code} = {meaning}" in _line(name), (name, code)
    for code, meaning in LABEL_MEANINGS.items():
        assert f"; {code} = {meaning}" in _line(LABEL)
    for bound in RANGES:
        assert _line(bound.field).lower().count(UNITS[bound.unit].lower()) == 1, bound.field
    assert "plausible" not in SYSTEM.replace("Every value must be plausible", "")
    assert _line("age") == "- age: age in years"
    assert "3 = normal; 6 = fixed defect; 7 = reversable defect" in _line("thal")
    assert all(f"; {code} = " in _line("cp") for code in range(1, 5))
    assert _line("chol") == "- chol: serum cholestoral in mg/dl"
    assert _line("trestbps") == (
        "- trestbps: resting blood pressure (in mm Hg on admission to the hospital)"
    )
    assert _line("thalach") == "- thalach: maximum heart rate achieved; beats per minute"
    assert _line("oldpeak") == ("- oldpeak: ST depression induced by exercise relative to rest; mm")


def test_the_system_prompt_names_the_fractional_fields_from_the_frame():
    assert "Whole numbers except oldpeak, which has one decimal place." in SYSTEM
    one = system(dataclasses.replace(UCI_FRAME, fractional=("chol",)))
    assert "Whole numbers except chol, which has one decimal place." in one
    two = system(dataclasses.replace(UCI_FRAME, fractional=("oldpeak", "chol")))
    assert "Whole numbers except oldpeak or chol, which have one decimal place." in two


EXPECTED_UCI_SYSTEM = (
    "You write synthetic patient records for a study of heart-disease prediction. "
    "Each record is one adult patient referred for coronary angiography. Invent new "
    "patients; do not reproduce records from any published dataset.\n"
    "\n"
    "Each record has these fields:\n"
    "- age: age in years\n"
    "- sex: sex; 0 = female; 1 = male\n"
    "- cp: chest pain type; 1 = typical angina; 2 = atypical angina; 3 = non-anginal "
    "pain; 4 = asymptomatic\n"
    "- trestbps: resting blood pressure (in mm Hg on admission to the hospital)\n"
    "- chol: serum cholestoral in mg/dl\n"
    "- fbs: fasting blood sugar > 120 mg/dl; 0 = fasting blood sugar <= 120 mg/dl; 1 "
    "= fasting blood sugar > 120 mg/dl\n"
    "- restecg: resting electrocardiographic results; 0 = normal; 1 = ST-T wave "
    "abnormality; 2 = left ventricular hypertrophy\n"
    "- thalach: maximum heart rate achieved; beats per minute\n"
    "- exang: exercise induced angina; 0 = no exercise induced angina; 1 = exercise "
    "induced angina\n"
    "- oldpeak: ST depression induced by exercise relative to rest; mm\n"
    "- slope: the slope of the peak exercise ST segment; 1 = upsloping; 2 = flat; 3 "
    "= downsloping\n"
    "- ca: number of major vessels (0-3) colored by flourosopy; 0 = 0 major vessels; "
    "1 = 1 major vessel; 2 = 2 major vessels; 3 = 3 major vessels\n"
    "- thal: thallium scintigraphy result; 3 = normal; 6 = fixed defect; 7 = "
    "reversable defect\n"
    "- target: diagnosis of heart disease (angiographic disease status); 0 = < 50% "
    "diameter narrowing; 1 = > 50% diameter narrowing\n"
    "\n"
    "Whole numbers except oldpeak, which has one decimal place. Every value must be "
    "plausible for a real patient."
)
"""`system(UCI_FRAME)` as `origin/main` rendered it before the opening came from the frame."""


def test_uci_system_text_is_unchanged():
    assert system(UCI_FRAME) == EXPECTED_UCI_SYSTEM


def test_a_frame_with_no_fractional_field_renders():
    text = system(dataclasses.replace(TOY, fractional=()))
    closing = text.splitlines()[-1]
    assert closing.startswith("Every field is a whole number.")
    assert "Whole numbers except" not in text


def test_the_opening_names_the_frames_study_and_cohort():
    opening = system(TOY).splitlines()[0]
    assert opening.startswith(
        f"You write synthetic patient records for a study of {TOY.study}. "
        f"Each record is one {TOY.cohort}. "
    )
    assert "heart-disease" not in opening


def test_the_system_prompt_asks_for_new_patients():
    assert "Invent new patients" in SYSTEM
    assert "do not reproduce records from any published dataset" in SYSTEM


@pytest.mark.parametrize(
    ("cell", "sex", "label", "sentence"),
    [
        (GUIDANCE.cell, 0, 1, "sex 0 (female), target 1 (> 50% diameter narrowing), age 35 to 54"),
        (OLDER, 1, 0, "sex 1 (male), target 0 (< 50% diameter narrowing), age 55 to 70"),
    ],
)
def test_the_cell_level_states_sex_label_age_band_record_count_and_row_count(
    cell, sex, label, sentence
):
    interval = intervals(cell, TRAIN_A, frame=UCI_FRAME)
    guidance = Guidance(cell=cell, sex=sex, label=label, records=4, interval=interval)
    system, user = render(guidance, 3, Level.CELL, frame=UCI_FRAME)
    assert system == SYSTEM
    assert "Write exactly 3 records" in user
    assert sentence in user
    assert "The training data holds 4 records of this group." in user
    low, high = guidance.interval["age"]
    assert f"Keep each numeric field within: age {low:g} to {high:g} years;" in user
    assert "oldpeak 0 to 2.7 mm" in user
    assert COHERENCE not in user


def test_the_coherent_level_adds_only_the_coherence_sentence():
    _, cell = render(GUIDANCE, 2, Level.CELL, frame=UCI_FRAME)
    _, coherent = render(GUIDANCE, 2, Level.COHERENT, frame=UCI_FRAME)
    assert coherent.startswith(cell)
    assert coherent.removeprefix(cell).strip() == COHERENCE


def test_intervals_clip_age_to_the_band_within_the_training_range():
    young = intervals(GUIDANCE.cell, TRAIN_A, frame=UCI_FRAME)
    assert young["age"] == (35, 54)
    found = ranges(TRAIN_A, UCI_FRAME)
    assert {name: bounds for name, bounds in young.items() if name != "age"} == {
        name: bounds for name, bounds in found.items() if name != "age"
    }
    assert intervals(OLDER, TRAIN_A, frame=UCI_FRAME)["age"] == (55, 70)
    frame = dataclasses.replace(UCI_FRAME, edges=(40, 60))
    assert intervals(("40-59", "male", 1), TRAIN_A, frame=frame)["age"] == (40, 59)
    assert intervals(("<40", "male", 1), TRAIN_A, frame=frame)["age"] == (35, 39)
    with pytest.raises(ValueError, match="60-69"):
        intervals(("60-69", "male", 1), TRAIN_A, frame=UCI_FRAME)


def test_rendering_is_deterministic():
    assert render(GUIDANCE, 2, Level.SUMMARY, frame=UCI_FRAME) == render(
        GUIDANCE, 2, Level.SUMMARY, frame=UCI_FRAME
    )


def _row(concept: str, field: str) -> FieldMapping:
    """The mapping table's row of `concept` that reads `field`."""
    (row,) = (
        row
        for row in FIELD_MAPPINGS
        if row.concept == concept and any(clause.field == field for clause in row.when)
    )
    return row


ANGINA_HISTORY = _row("HP:0001681", "cp")
ANGINA_EXERTION = _row("HP:0001681", "exang")
CHOLESTEROL = _row("HP:0003124", "chol")
ST_DEPRESSION = _row("HP:0012250", "oldpeak")
CALCIFICATION = _row("HP:0001717", "ca")

CHAIN = (
    Relationship(Statement("hpo", "HP:0001681", "HP:0011675"), "Angina pectoris", "Arrhythmia"),
    Relationship(Statement("hpo", "HP:0011675", "HP:0001626"), "Arrhythmia", "Heart disease"),
)
"""Two statements with invented parents: the renderer takes labels as given."""

FINDINGS = (
    Finding((ANGINA_HISTORY, ANGINA_EXERTION), CHAIN),
    Finding((CHOLESTEROL,), (CHAIN[1],)),
    Finding((ST_DEPRESSION, CALCIFICATION), (CHAIN[0],)),
)

FINDING_LINES = (
    "- cp is 1 (typical angina) or exang is 1 (exercise induced angina) (Angina pectoris)",
    "- chol is at least 240 mg/dL (Hypercholesterolemia)",
    (
        "- oldpeak is at least 1.0 mm and slope is 2 or 3 (flat, downsloping) or ca is 1, 2 or 3 "
        "(1 major vessel, 2 major vessels, 3 major vessels) (ST segment depression, Coronary "
        "artery calcification)"
    ),
)

CHAINED_LINES = (
    (
        f"{FINDING_LINES[0]}: Angina pectoris is a kind of Arrhythmia; Arrhythmia is a kind "
        "of Heart disease"
    ),
    f"{FINDING_LINES[1]}: Arrhythmia is a kind of Heart disease",
    f"{FINDING_LINES[2]}: Angina pectoris is a kind of Arrhythmia",
)
"""`FINDING_LINES` with their `is_a` chains, as the ontology level renders them."""

SENTENCES = {
    Basis.ABSENT: (
        "No record of this group in the training data shows the findings below. At least one "
        "record should show each; otherwise keep each as frequent as is clinically plausible for "
        "this group."
    ),
    Basis.RARE: (
        "The findings below are rarer in this group than among comparable patients. Show each "
        "about as often as among comparable patients, as far as is clinically plausible for this "
        "group."
    ),
    Basis.SHOWN: (
        "Records of this group show the findings below; not every record shows each. Keep each "
        "as frequent as is clinically plausible for this group."
    ),
}


def _found(basis: Basis) -> Guidance:
    return dataclasses.replace(GUIDANCE, basis=basis, findings=FINDINGS)


def test_the_findings_level_renders_field_states_with_code_meanings_units_and_concept_labels():
    _, coherent = render(_found(Basis.ABSENT), 2, Level.COHERENT, frame=UCI_FRAME)
    _, findings = render(_found(Basis.ABSENT), 2, Level.FINDINGS, frame=UCI_FRAME)
    assert findings.startswith(coherent)
    added = findings.removeprefix(coherent).strip()
    assert added.splitlines() == [SENTENCES[Basis.ABSENT], *FINDING_LINES]


@pytest.mark.parametrize("basis", list(SENTENCES))
def test_each_basis_has_its_own_wording(basis):
    _, user = render(_found(basis), 2, Level.FINDINGS, frame=UCI_FRAME)
    assert SENTENCES[basis] in user
    assert [other for other in SENTENCES.values() if other in user] == [SENTENCES[basis]]


def test_the_ontology_level_adds_each_findings_is_a_chain():
    _, findings = render(_found(Basis.SHOWN), 2, Level.FINDINGS, frame=UCI_FRAME)
    _, ontology = render(_found(Basis.SHOWN), 2, Level.ONTOLOGY, frame=UCI_FRAME)
    assert findings.splitlines()[-3:] == list(FINDING_LINES)
    assert ontology.splitlines() == [*findings.splitlines()[:-3], *CHAINED_LINES]


def test_basis_none_omits_the_findings_block():
    _, coherent = render(_found(Basis.NONE), 2, Level.COHERENT, frame=UCI_FRAME)
    for level in (Level.FINDINGS, Level.ONTOLOGY):
        assert render(_found(Basis.NONE), 2, level, frame=UCI_FRAME)[1] == coherent
    shown_without = dataclasses.replace(GUIDANCE, basis=Basis.SHOWN)
    assert render(shown_without, 2, Level.ONTOLOGY, frame=UCI_FRAME)[1] == coherent


def test_the_is_a_chain_reads_from_the_lowest_class_up_whatever_the_stored_order():
    top_down = dataclasses.replace(
        GUIDANCE, basis=Basis.SHOWN, findings=(Finding((ANGINA_HISTORY,), CHAIN[::-1]),)
    )
    _, user = render(top_down, 2, Level.ONTOLOGY, frame=UCI_FRAME)
    assert user.splitlines()[-1] == (
        "- cp is 1 (typical angina) (Angina pectoris): Angina pectoris is a kind of Arrhythmia; "
        "Arrhythmia is a kind of Heart disease"
    )


def test_a_cyclic_chain_is_rendered_whole_in_stored_order():
    back = Relationship(
        Statement("hpo", "HP:0011675", "HP:0001681"), "Arrhythmia", "Angina pectoris"
    )
    cyclic = dataclasses.replace(
        GUIDANCE, basis=Basis.SHOWN, findings=(Finding((ANGINA_HISTORY,), (CHAIN[0], back)),)
    )
    _, user = render(cyclic, 2, Level.ONTOLOGY, frame=UCI_FRAME)
    assert user.splitlines()[-1].endswith(
        ": Angina pectoris is a kind of Arrhythmia; Arrhythmia is a kind of Angina pectoris"
    )


PROFILE = Profile(
    40,
    (
        Share("HP:0001681", "Angina pectoris", (ANGINA_HISTORY, ANGINA_EXERTION), 1 / 3),
        Share("HP:0003124", "Hypercholesterolemia", (CHOLESTEROL,), 0.5),
        Share("HP:0012250", "ST segment depression", (ST_DEPRESSION,), 0.0),
    ),
)

PROFILE_BLOCK = (
    (
        "Among comparable patients (the same target, other age and sex groups; 40 records), these "
        "findings occur at these rates:"
    ),
    "- Angina pectoris (cp is 1 (typical angina) or exang is 1 (exercise induced angina)): 33%",
    "- Hypercholesterolemia (chol is at least 240 mg/dL): 50%",
    (
        "- ST segment depression (oldpeak is at least 1.0 mm and slope is 2 or 3 (flat, "
        "downsloping)): 0%"
    ),
)

SUMMARY = {
    "age": Spread(47.4, 55.0, 62.6),
    "trestbps": Spread(120.0, 130.2, 140.0),
    "chol": Spread(211.25, 241.0, 275.75),
    "thalach": Spread(130.0, 145.6, 160.0),
    "oldpeak": Spread(0.26, 1.04, 1.96),
}

SUMMARY_BLOCK = (
    "Among the same patients, numeric fields run as follows (median; middle half):",
    "- age: 55; 47 to 63 years",
    "- trestbps: 130; 120 to 140 mm Hg",
    "- chol: 241; 211 to 276 mg/dL",
    "- thalach: 146; 130 to 160 beats per minute",
    "- oldpeak: 1.0; 0.3 to 2.0 mm",
)

FULL = dataclasses.replace(
    GUIDANCE, basis=Basis.ABSENT, findings=FINDINGS, profile=PROFILE, summary=SUMMARY
)


def test_the_profile_level_adds_shares_as_whole_percents_with_their_field_states():
    _, ontology = render(FULL, 2, Level.ONTOLOGY, frame=UCI_FRAME)
    _, profile = render(FULL, 2, Level.PROFILE, frame=UCI_FRAME)
    assert profile == "\n\n".join([ontology, "\n".join(PROFILE_BLOCK)])


def test_the_summary_level_adds_median_and_middle_half_with_units():
    _, profile = render(FULL, 2, Level.PROFILE, frame=UCI_FRAME)
    _, summary = render(FULL, 2, Level.SUMMARY, frame=UCI_FRAME)
    assert summary == "\n\n".join([profile, "\n".join(SUMMARY_BLOCK)])


def test_each_level_adds_exactly_its_own_block():
    texts = [render(FULL, 2, level, frame=UCI_FRAME)[1].split("\n\n") for level in LEVELS]
    for before, after, level in zip(texts, texts[1:], LEVELS[1:], strict=False):
        if level is Level.ONTOLOGY:
            assert after[:-1] == before[:-1], level
            assert " is a kind of " not in before[-1], level
            assert after[-1].splitlines() == [SENTENCES[Basis.ABSENT], *CHAINED_LINES], level
        else:
            assert after[:-1] == before, level
    bare = dataclasses.replace(GUIDANCE, profile=None, summary=None)
    assert render(bare, 2, Level.SUMMARY, frame=UCI_FRAME) == render(
        bare, 2, Level.COHERENT, frame=UCI_FRAME
    )


def shown_rows(count: int) -> tuple[dict[str, int | float], ...]:
    rows = TRAIN_A.loc[:, list(UCI_FRAME.columns)].head(count)
    return tuple(
        {name: (float(value) if name in UCI_FRAME.numeric else int(value)) for name, value in row}
        for row in (
            zip(rows.columns, values, strict=True) for values in rows.itertuples(index=False)
        )
    )


def test_render_shown_opens_with_the_render_sentence_and_lists_json_lines():
    shown = shown_rows(3)
    system_text, user = render_shown(GUIDANCE, 5, shown, frame=UCI_FRAME)
    assert system_text == SYSTEM
    assert PROMPT.match(user)
    assert RECORD_LINE.search(user) is None
    listed = [json.loads(line) for line in user.splitlines() if line.startswith("{")]
    assert listed == list(shown)
    assert all(tuple(record) == UCI_FRAME.columns for record in listed)
    first = render(GUIDANCE, 5, Level.CELL, frame=UCI_FRAME)[1].split("\n\n")
    assert user.split("\n\n")[:2] == first[:2]
