"""The consistency checks: one chain per field, each planted record failing its check alone."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import CATEGORICAL, FRAME_COLUMNS
from bko.ontology.mapping import RANGES
from bko.validate.checks import CHECKS, failures
from tests._baselines_fixtures import EVALUATION
from tests._validate_fixtures import BASE, PLANTED, planted


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

MEASURED = ("age", "trestbps", "chol", "thalach", "oldpeak")
CODEBOOK = {
    "sex": (0, 1),
    "cp": (1, 2, 3, 4),
    "fbs": (0, 1),
    "restecg": (0, 1, 2),
    "exang": (0, 1),
    "slope": (1, 2, 3),
    "ca": (0, 1, 2, 3),
    "thal": (3, 6, 7),
    "target": (0, 1),
}
HOSTILE = {
    "none": None,
    "empty": "",
    "text": "abc",
    "numeric-text": "63",
    "list": [63],
    "dict": {},
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
    "true": True,
    "false": False,
    "numpy-bool": np.True_,
    "numpy-nan": np.float64("nan"),
    "numpy-float32": np.float32(2.3),
}
"""What JSON can decode to that is not a finite number, plus numpy's bool, NaN and a float narrower
than float64, since numpy integers and float64 are accepted."""


def test_base_is_the_first_real_record():
    assert EVALUATION.to_dict("records")[0] == dict(BASE)


def test_the_base_record_passes():
    assert failures(BASE) == ()


def test_the_checks_are_the_field_set_then_each_fields_number_bound_and_decimals():
    expected = {
        "fields",
        *(f"number:{name}" for name in BASE),
        *(f"code:{name}" for name in CODEBOOK),
        *(f"range:{name}" for name in MEASURED),
        *(f"precision:{name}" for name in MEASURED),
    }
    assert set(CHECKS) == expected
    assert len(CHECKS) == len(expected)
    assert CHECKS[0] == "fields"


def test_the_codebook_and_the_ranges_bound_every_frame_field_once():
    coded = {*CATEGORICAL, LABEL}
    ranged = {row.field for row in RANGES}
    assert not coded & ranged
    assert coded | ranged == set(FRAME_COLUMNS)


@pytest.mark.parametrize("check", list(PLANTED), ids=list(PLANTED))
def test_each_planted_record_fails_its_check_alone(check):
    assert failures(PLANTED[check]) == (check,)


def test_every_check_has_a_planted_record():
    assert set(PLANTED) == set(CHECKS)


@pytest.mark.parametrize("row", RANGES, ids=[row.field for row in RANGES])
def test_a_range_admits_both_bounds(row):
    assert failures(planted(**{row.field: row.low})) == ()
    assert failures(planted(**{row.field: row.high})) == ()


@pytest.mark.parametrize("row", RANGES, ids=[row.field for row in RANGES])
def test_one_step_past_either_bound_fails_the_range_alone(row):
    step = 0.1 if row.field == "oldpeak" else 1
    for value in (row.low - step, row.high + step):
        assert failures(planted(**{row.field: value})) == (f"range:{row.field}",)


def test_a_hair_past_a_bound_fails_the_range():
    assert failures(planted(oldpeak=math.nextafter(10.0, math.inf))) == ("range:oldpeak",)


@pytest.mark.parametrize(
    ("field", "code"),
    [(field, code) for field, codes in CODEBOOK.items() for code in codes],
)
def test_every_code_passes_as_an_int_or_a_whole_float(field, code):
    assert failures(planted(**{field: code})) == ()
    assert failures(planted(**{field: float(code)})) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cp", 5),
        ("ca", -1),
        ("slope", 4),
        ("thal", 0),
        ("thal", 5),
        ("thal", 8),
        ("target", -1),
        ("cp", 1.5),
        ("target", 0.5),
    ],
)
def test_a_value_outside_its_codebook_fails_the_code_alone(field, value):
    assert failures(planted(**{field: value})) == (f"code:{field}",)


@pytest.mark.parametrize("value", list(HOSTILE.values()), ids=list(HOSTILE))
def test_a_value_that_is_not_a_finite_number_fails_its_number_check_alone(value):
    for name in BASE:
        assert failures(planted(**{name: value})) == (f"number:{name}",)


def test_a_missing_field_fails_its_number_check_alone():
    for name in BASE:
        candidate = {key: value for key, value in BASE.items() if key != name}
        assert failures(candidate) == (f"number:{name}",)


@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["huge", "negative-huge"])
def test_an_integer_too_large_for_a_float_fails_its_bound_and_never_raises(value):
    assert failures(planted(chol=value)) == ("range:chol",)
    assert failures(planted(ca=value)) == ("code:ca",)


def test_numpy_numbers_pass():
    assert failures(planted(age=np.int32(63), thal=np.int64(6), oldpeak=np.float64(2.3))) == ()


def test_whole_floats_pass_whole_fields():
    assert failures(planted(age=63.0, chol=233.0, ca=0.0, target=0.0)) == ()


def test_a_bound_failure_stops_the_fields_chain():
    assert failures(planted(chol=5.2)) == ("range:chol",)


@pytest.mark.parametrize(
    "candidate",
    [None, 63, "age", ["age"], ("age", 63), list(BASE.items())],
    ids=["none", "int", "str", "list", "pair", "items"],
)
def test_a_candidate_that_is_not_a_mapping_fails_the_field_set_alone(candidate):
    assert failures(candidate) == ("fields",)


def test_an_extra_key_fails_the_field_set_and_the_fields_are_still_checked():
    assert failures(planted(num=0, chol=601)) == ("fields", "range:chol")
    assert failures({**BASE, 0: 1}) == ("fields",)


def test_a_misspelt_key_fails_the_field_set_and_its_fields_number_check():
    candidate = {("Age" if key == "age" else key): value for key, value in BASE.items()}
    assert failures(candidate) == ("fields", "number:age")


def test_several_defects_are_reported_once_each_in_check_order():
    got = failures(planted(age=200, cp=9, trestbps="high", num=2))
    assert got == ("fields", "number:trestbps", "range:age", "code:cp")
