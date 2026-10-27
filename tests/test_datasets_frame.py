"""The frame value: UCI's holds UCI's constants, and a frame refuses values that contradict it."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import RANGES, UCI_FRAME
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
