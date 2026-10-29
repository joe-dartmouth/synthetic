"""A prescribing request: per asked record, the concepts it must show, rendered and checked."""

from __future__ import annotations

import dataclasses
from typing import Final

import pytest

from bko.baselines.records import Drop, Prescribed, assess, drops
from bko.baselines.request import Request, Strata
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.rehearse import PROMPT
from bko.generate.calls import request
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Level
from bko.generate.prompt import render
from bko.generate.skeleton import prescribable, skeleton
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import shown
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import GUIDANCE
from tests._validate_fixtures import BASE, planted
from tests.test_generate_direct import frame
from tests.test_generate_prompt import FULL


BASE_SHOWS: Final = frozenset(
    {
        "HP:0001681",
        "HP:0100749",
        "HP:0004421",
        "HP:0003074",
        "HP:0003115",
        "HP:0001712",
        "HP:0012250",
        "HP:0500018",
    }
)
"""The concepts `BASE`, the first Cleveland record with target 0, shows."""

LINES: Final = (
    frozenset({"HP:0100749", "HP:0001681", "HP:0003124"}),
    frozenset(),
    frozenset({"HP:0012250", "HP:0003115", "HP:0001712"}),
)
"""A 3-record draw, label-only concepts left out."""

SKELETON: Final = skeleton(LINES, FIELD_MAPPINGS)


def _record(**changes: float) -> dict[str, float]:
    """`BASE` with `changes`, typed as `FieldMapping.applies` reads a record."""
    return {**BASE, **changes}


def test_shown_names_the_concepts_a_record_shows():
    assert shown(BASE, FIELD_MAPPINGS) == BASE_SHOWS
    sick = _record(target=1, cp=4, exang=0)
    assert shown(sick, FIELD_MAPPINGS) == (BASE_SHOWS - {"HP:0001681", "HP:0100749"}) | {
        "MONDO:0006715",
        "HP:0005145",
    }
    assert shown(BASE, ()) == frozenset()


def test_skeleton_leaves_out_label_only_concepts_and_refuses_an_unnamed_one():
    named = [one.concept for one in SKELETON.named]
    assert named == sorted(named)
    assert len(named) == 10
    assert {"MONDO:0006715", "HP:0005145"}.isdisjoint(named)
    angina = next(one for one in SKELETON.named if one.concept == "HP:0001681")
    assert angina.label == "Angina pectoris"
    assert [row.when[0].field for row in angina.states] == ["cp", "exang"]
    assert {"MONDO:0006715", "HP:0005145"}.isdisjoint(
        row.concept for row in prescribable(FIELD_MAPPINGS)
    )
    assert SKELETON.shows(_record(target=1)) == BASE_SHOWS
    with pytest.raises(ValueError, match=r"line 2 names \['MONDO:0006715'\]"):
        skeleton((frozenset(), frozenset({"MONDO:0006715"})), FIELD_MAPPINGS)


EXPECTED_USER: Final = (
    "Write exactly 3 records of patients in this group: sex 0 (female), target 1 (> 50% diameter "
    "narrowing), age 35 to 54 years. The training data holds 2 records of this group.\n"
    "\n"
    "Keep each numeric field within: age 35 to 54 years; trestbps 110 to 155 mm Hg; chol 180 to "
    "300 mg/dL; thalach 120 to 174 beats per minute; oldpeak 0 to 2.7 mm.\n"
    "\n"
    "Keep each record clinically coherent: its values should fit the patient's disease status, "
    "age and sex together.\n"
    "\n"
    "Each record below must show exactly the listed findings and no other finding from this list; "
    "these lines take precedence over any finding named above.\n"
    "- Angina pectoris (cp is 1 (typical angina) or exang is 1 (exercise induced angina))\n"
    "- Left ventricular hypertrophy (restecg is 2 (left ventricular hypertrophy))\n"
    "- Coronary artery calcification (ca is 1, 2 or 3 (1 major vessel, 2 major vessels, 3 major "
    "vessels))\n"
    "- Hyperglycemia (fbs is 1 (fasting blood sugar > 120 mg/dl))\n"
    "- Abnormal EKG (restecg is 1 or 2 (ST-T wave abnormality, left ventricular hypertrophy))\n"
    "- Hypercholesterolemia (chol is at least 240 mg/dL)\n"
    "- Elevated systolic blood pressure (trestbps is at least 130 mm Hg)\n"
    "- ST segment depression (oldpeak is at least 1.0 mm and slope is 2 or 3 (flat, "
    "downsloping))\n"
    "- Chest pain (cp is 1, 2 or 3 (typical angina, atypical angina, non-anginal pain))\n"
    "- Abnormal cardiac exercise stress test (thal is 6 or 7 (fixed defect, reversable defect))\n"
    "Record 1: Angina pectoris; Hypercholesterolemia; Chest pain\n"
    "Record 2: none of these findings\n"
    "Record 3: Left ventricular hypertrophy; Abnormal EKG; ST segment depression"
)
"""`LINES` at `summary` for `GUIDANCE`: concepts and each record's labels sorted by curie."""


def test_the_user_turn_with_one_line_per_record():
    _, user = render(GUIDANCE, 3, Level.SUMMARY, skeleton=SKELETON)
    assert user == EXPECTED_USER
    _, bare = render(FULL, 3, Level.SUMMARY)
    _, full = render(FULL, 3, Level.SUMMARY, skeleton=SKELETON)
    assert full.startswith(f"{bare}\n\n")
    block = full.removeprefix(f"{bare}\n\n").splitlines()
    assert block[0].startswith("Each record below must show exactly the listed findings")
    assert len(block) == 1 + 10 + 3
    assert block == EXPECTED_USER.split("\n\n")[-1].splitlines()


def test_the_prompt_regex_still_reads_a_skeleton_user_turn():
    _, user = render(FULL, 3, Level.SUMMARY, skeleton=SKELETON)
    found = PROMPT.match(user)
    assert found is not None
    assert (found["rows"], found["sex"], found["label"]) == ("3", "0", "1")


def test_request_refuses_lines_that_do_not_match_rows(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=lambda: FIXED_NOW, frame="uci") as log:
        with pytest.raises(ValueError, match="2 lines for 3 records"):
            request(GUIDANCE, 3, log=log, skeleton=dataclasses.replace(SKELETON, lines=LINES[:2]))
        assert log.calls == 0
        pending = request(GUIDANCE, 3, log=log, skeleton=SKELETON)
    assert pending.call.user == render(GUIDANCE, 3, Level.SUMMARY, skeleton=SKELETON)[1]
    assert sum('"kind": "request"' in line for line in path.read_text().splitlines()) == 1


CELL: Final = ("<55", "female", 1)


def test_drops_a_record_whose_concepts_differ_after_cell_before_quota():
    train = frame([planted(age=40, sex=0, target=1), planted(age=70, sex=1, target=0)])
    rows = frame(
        [
            planted(age=40, sex=0, target=1),
            planted(age=41, sex=0, target=1, cp=4),
            planted(age=41, sex=0, target=1, chol=999),
            planted(age=63, sex=0, target=1),
            planted(age=42, sex=0, target=1),
        ]
    )
    asked = Request(quota={CELL: 1}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    one, other = frozenset({"a"}), frozenset({"b"})
    lined = Prescribed(CELL, (one, one), (one, other, one, one, one))
    found = assess(rows, train, asked, prescribed=lined)
    assert found.reasons == (None, Drop.CONCEPTS, Drop.RANGE, Drop.CELL, Drop.QUOTA)
    assert found.lines == (0, None, None, None, None)
    assert drops(rows, train, asked, prescribed=lined) == found.reasons
    assert drops(rows, train, asked) == (None, Drop.QUOTA, Drop.RANGE, Drop.CELL, Drop.QUOTA)
    assert list(Drop) == [Drop.RANGE, Drop.CELL, Drop.CONCEPTS, Drop.QUOTA]
    with pytest.raises(ValueError, match="2 concept sets for 5 rows"):
        drops(rows, train, asked, prescribed=dataclasses.replace(lined, shows=(one, one)))


def test_a_record_matches_any_line_of_its_call_not_consumed_in_position_order():
    train = frame([planted(age=40, sex=0, target=1), planted(age=70, sex=1, target=0)])
    rows = frame([planted(age=40 + n, sex=0, target=1) for n in range(4)])
    asked = Request(quota={CELL: 4}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    a, b, c = frozenset({"a"}), frozenset({"b"}), frozenset()
    reordered = assess(rows, train, asked, prescribed=Prescribed(CELL, (a, b, c, a), (c, a, a, b)))
    assert reordered.reasons == (None, None, None, None)
    assert reordered.lines == (2, 0, 3, 1)
    skipped = assess(rows[:3], train, asked, prescribed=Prescribed(CELL, (a, b, c, a), (b, c, b)))
    assert skipped.reasons == (None, None, Drop.CONCEPTS)
    assert skipped.lines == (1, 2, None)


def test_a_prescribed_record_kept_in_another_asked_cell_is_dropped_for_its_cell():
    train = frame([planted(age=40, sex=0, target=1), planted(age=70, sex=1, target=0)])
    rows = frame([planted(age=63, sex=0, target=1), planted(age=40, sex=0, target=1)])
    sibling = (">=55", "female", 1)
    asked = Request(quota={CELL: 2, sibling: 1}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    line = frozenset({"a"})
    found = assess(rows, train, asked, prescribed=Prescribed(CELL, (line, line), (line, line)))
    assert found.reasons == (Drop.CELL, None)
    assert found.lines == (None, 0)
    assert drops(rows, train, asked) == (None, None)
