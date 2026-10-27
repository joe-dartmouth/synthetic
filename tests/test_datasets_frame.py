"""The frame value, and validation and generation following the frame they are given."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import RANGES, UCI_FRAME
from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import SETTINGS, Call, Level, record_schema
from bko.generate.prompt import SYSTEM, render, system
from bko.validate.checks import CHECKS, checks, validate
from tests._frame_fixtures import TOY


def test_the_uci_frame_holds_the_uci_values():
    assert UCI_FRAME.name == "uci"
    assert UCI_FRAME.features == (
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    )
    assert UCI_FRAME.numeric == ("age", "trestbps", "chol", "thalach", "oldpeak")
    assert UCI_FRAME.fractional == ("oldpeak",)
    assert UCI_FRAME.fraction_digits == 1
    assert UCI_FRAME.categorical == ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
    assert UCI_FRAME.codebook == {
        "sex": {0: "female", 1: "male"},
        "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
        "fbs": {0: "fasting blood sugar <= 120 mg/dl", 1: "fasting blood sugar > 120 mg/dl"},
        "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
        "exang": {0: "no exercise induced angina", 1: "exercise induced angina"},
        "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
        "ca": {
            0: "0 major vessels",
            1: "1 major vessel",
            2: "2 major vessels",
            3: "3 major vessels",
        },
        "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},
    }
    assert UCI_FRAME.descriptions == {
        "age": "age in years",
        "sex": "sex",
        "cp": "chest pain type",
        "trestbps": "resting blood pressure (in mm Hg on admission to the hospital)",
        "chol": "serum cholestoral in mg/dl",
        "fbs": "fasting blood sugar > 120 mg/dl",
        "restecg": "resting electrocardiographic results",
        "thalach": "maximum heart rate achieved",
        "exang": "exercise induced angina",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "the slope of the peak exercise ST segment",
        "ca": "number of major vessels (0-3) colored by flourosopy",
        "thal": "thallium scintigraphy result",
        "target": "diagnosis of heart disease (angiographic disease status)",
    }
    assert UCI_FRAME.ranges is RANGES
    assert [(r.field, r.unit, r.low, r.high) for r in UCI_FRAME.ranges] == [
        ("age", "a", 18, 120),
        ("trestbps", "mm[Hg]", 10, 250),
        ("chol", "mg/dL", 32, 600),
        ("thalach", "/min", 10, 500),
        ("oldpeak", "mm", -10, 10),
    ]
    assert UCI_FRAME.label == "target"
    assert UCI_FRAME.label_codes == (0, 1)
    assert UCI_FRAME.label_meanings == {
        0: "< 50% diameter narrowing",
        1: "> 50% diameter narrowing",
    }
    assert UCI_FRAME.edges == (55,)


def test_the_uci_frame_derives_the_evaluation_columns_and_dtypes():
    assert UCI_FRAME.columns == (*UCI_FRAME.features, "target")
    assert dict(UCI_FRAME.dtypes) == {
        name: "float64" if name == "oldpeak" else "int64" for name in UCI_FRAME.columns
    }


def test_a_toy_frame_with_other_edges_builds():
    assert TOY.columns == ("age", "sex", "level", "target")
    assert dict(TOY.dtypes) == {
        "age": "int64",
        "sex": "int64",
        "level": "float64",
        "target": "int64",
    }
    assert TOY.edges == (45, 70)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"numeric": ("age", "level", "weight")}, "numeric field weight is not a feature"),
        ({"fractional": ("sex",)}, "fractional field sex is not numeric"),
        ({"ranges": TOY.ranges[:1]}, "numeric field level has 0 ranges, not 1"),
        ({"ranges": (*TOY.ranges, TOY.ranges[1])}, "numeric field level has 2 ranges, not 1"),
        ({"ranges": (*TOY.ranges, replace(TOY.ranges[0], field="sex"))}, "sex is not numeric"),
        ({"descriptions": {"age": "a", "sex": "s", "level": "l"}}, "column target has no desc"),
    ],
    ids=["numeric", "fractional", "no-range", "two-ranges", "stray-range", "description"],
)
def test_a_frame_refuses_values_that_contradict_it(change, match):
    with pytest.raises(ValueError, match=match):
        replace(TOY, **change)


def test_frames_compare_by_identity():
    assert replace(TOY) != TOY
    assert isinstance(hash(TOY), int)
    assert isinstance(UCI_FRAME, FrameSpec)


TOY_RECORD = {"age": 50, "sex": 0, "level": 1.25, "target": 1}
TOY_CELL = ("45-69", "female", 1)


def test_validation_reads_the_toy_frame():
    planted = [TOY_RECORD, TOY_RECORD | {"level": 5.0}, TOY_RECORD | {"level": 1.255}]
    result = validate(planted, frame=TOY)
    assert result.accepted.columns.tolist() == ["age", "sex", "level", "target"]
    assert result.accepted.dtypes.astype(str).to_dict() == dict(TOY.dtypes)
    assert [rejection.failed for rejection in result.rejected] == [
        ("range:level",),
        ("precision:level",),
    ]
    assert list(result.rejections) == list(checks(TOY))
    assert result.rejections["range:level"] == 1


def test_validation_refuses_a_uci_field_under_the_toy_frame():
    result = validate([TOY_RECORD | {"cp": 1}], frame=TOY)
    assert result.rejected[0].failed == ("fields",)


def test_the_checks_follow_the_frame():
    assert checks(TOY) == (
        "fields",
        "number:age",
        "number:sex",
        "number:level",
        "number:target",
        "range:age",
        "precision:age",
        "code:sex",
        "range:level",
        "precision:level",
        "code:target",
    )
    assert checks(UCI_FRAME) == CHECKS
    assert checks(TOY) is checks(TOY)


def test_the_record_schema_names_the_toy_columns():
    schema: Any = record_schema(0, 1, frame=TOY)
    record = schema["properties"]["records"]["items"]
    assert record["required"] == ["age", "sex", "level", "target"]
    assert record["properties"] == {
        "age": {"type": "integer"},
        "sex": {"type": "integer", "enum": [0]},
        "level": {"type": "number"},
        "target": {"type": "integer", "enum": [1]},
    }


def test_a_calls_params_carry_its_frames_schema():
    call = Call(SETTINGS, 0, 1, "system", "user", frame=TOY)
    params: Any = call.params()
    schema = params["output_config"]["format"]["schema"]
    assert schema == record_schema(0, 1, frame=TOY)
    assert Call(SETTINGS, 0, 1, "system", "user").frame is UCI_FRAME


def test_the_system_prompt_lists_the_toy_fields():
    text = system(TOY)
    assert "- age: age at the visit; years, plausible from 30 to 90" in text
    assert "- sex: recorded sex; 0 = female; 1 = male" in text
    assert "- level: toy marker level; mg/dL, plausible from 0.5 to 4.5" in text
    assert "- target: whether the toy condition is present; 0 = toy condition absent" in text
    assert "cp" not in text
    assert "Whole numbers except level, which has one decimal place." not in text
    assert "Whole numbers except level, which has 2 decimal places." in text


def test_the_uci_system_prompt_is_the_shared_one():
    assert system(UCI_FRAME) == SYSTEM
    assert system(UCI_FRAME) is system(UCI_FRAME)


def test_intervals_follow_the_toy_ranges_and_edges():
    assert intervals(TOY_CELL, frame=TOY) == {"age": (45, 69), "level": (0.5, 4.5)}
    assert intervals(("<45", "male", 0), frame=TOY) == {"age": (30, 44), "level": (0.5, 4.5)}
    with pytest.raises(ValueError, match="not one of"):
        intervals(("<55", "male", 0), frame=TOY)


def test_the_user_turn_reads_the_toy_frame():
    guidance = Guidance(
        cell=TOY_CELL, sex=0, label=1, records=3, interval=intervals(TOY_CELL, frame=TOY)
    )
    sent, user = render(guidance, 4, Level.CELL, frame=TOY)
    assert sent == system(TOY)
    assert user.startswith(
        "Write exactly 4 records of patients in this group: sex 0 (female), target 1 (toy "
        "condition present), age 45 to 69 years."
    )
    assert "Keep each numeric field within: age 45 to 69 years; level 0.5 to 4.5 mg/dL." in user
