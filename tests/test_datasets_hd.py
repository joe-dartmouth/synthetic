"""Lautrup et al.'s `hd` rows read as UCI records (ADR `research/syntheval-reading`)."""

from __future__ import annotations

import logging

import pytest

from bko.datasets.errors import ContractError
from bko.datasets.hd import (
    CP_CODES,
    HD_COLUMNS,
    MISSING_CA,
    MISSING_THAL,
    RESTECG_CODES,
    SLOPE_CODES,
    THAL_CODES,
    from_hd,
)
from bko.datasets.uci import CODEBOOK, FRAME_COLUMNS, FRAME_DTYPES


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

HEADER = ",".join(HD_COLUMNS)
"""The header line of their files."""


def hd_line(**fields: object) -> str:
    """One `hd` data line: a complete record, with `fields` replacing its values."""
    values: dict[str, object] = {
        "age": 58,
        "sex": 0,
        "cp": 3,
        "trestbps": 150,
        "chol": 283,
        "fbs": 1,
        "restecg": 0,
        "thalach": 162,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
        "target": 1,
    }
    values |= fields
    return ",".join(str(values[name]) for name in HD_COLUMNS)


def hd_text(*lines: str) -> str:
    """A whole `hd` file holding `lines` under the header."""
    return "\n".join((HEADER, *lines)) + "\n"


def test_their_columns_read_as_ours():
    maps = {"cp": CP_CODES, "restecg": RESTECG_CODES, "slope": SLOPE_CODES, "thal": THAL_CODES}
    lines = [hd_line(**{name: code}) for name, codes in maps.items() for code in codes]
    frame = from_hd(hd_text(*lines))
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert {name: str(dtype) for name, dtype in frame.dtypes.items()} == dict(FRAME_DTYPES)
    assert frame.index.name == "record_id"
    assert list(frame.index) == list(range(1, len(lines) + 1))
    expected = [ours for codes in maps.values() for ours in codes.values()]
    found = [
        int(frame.loc[row, name])
        for row, name in zip(
            frame.index, [name for name, codes in maps.items() for _ in codes], strict=True
        )
    ]
    assert found == expected
    for name, codes in maps.items():
        assert set(codes.values()) == set(CODEBOOK[name])
    assert frame.loc[1, "oldpeak"] == pytest.approx(1.0)
    assert frame.loc[1, "chol"] == 283


def test_the_maps_are_the_match_against_the_cleveland_file():
    assert CP_CODES == {0: 4, 1: 2, 2: 3, 3: 1}
    assert RESTECG_CODES == {0: 2, 1: 0, 2: 1}
    assert SLOPE_CODES == {0: 3, 1: 2, 2: 1}
    assert THAL_CODES == {1: 6, 2: 3, 3: 7}
    assert (MISSING_CA, MISSING_THAL) == (4, 0)


def test_the_target_is_inverted():
    frame = from_hd(hd_text(hd_line(target=1), hd_line(target=0)))
    assert list(frame["target"]) == [0, 1]


def test_rows_with_a_missing_code_are_dropped_and_counted(caplog):
    text = hd_text(hd_line(), hd_line(ca=MISSING_CA), hd_line(thal=MISSING_THAL), hd_line(age=61))
    with caplog.at_level(logging.INFO, logger="bko.datasets.hd"):
        frame = from_hd(text)
    assert list(frame.index) == [1, 4]
    assert list(frame["age"]) == [58, 61]
    lines = [record.getMessage() for record in caplog.records]
    assert lines == ["hd rows=4 kept=2 dropped=2 missing_ca=1 missing_thal=1"]


@pytest.mark.parametrize(
    ("name", "code"),
    [("cp", 4), ("restecg", 3), ("slope", 3), ("thal", 4), ("ca", 5), ("sex", 2), ("target", 2)],
)
def test_a_code_outside_the_map_is_refused(name, code):
    with pytest.raises(ContractError, match=name):
        from_hd(hd_text(hd_line(**{name: code})))


def test_a_file_with_other_columns_is_refused():
    with pytest.raises(ContractError, match="columns"):
        from_hd("age,sex\n58,0\n")


def test_a_value_that_is_not_a_number_is_refused():
    with pytest.raises(ContractError, match="chol"):
        from_hd(hd_text(hd_line(chol="high")))


def test_a_fraction_in_a_whole_field_is_refused():
    with pytest.raises(ContractError, match="age"):
        from_hd(hd_text(hd_line(age=58.5)))


def test_a_file_of_only_missing_codes_is_refused():
    with pytest.raises(ContractError, match="no complete"):
        from_hd(hd_text(hd_line(ca=MISSING_CA)))


@pytest.mark.parametrize(
    ("missing", "bad"),
    [({"ca": MISSING_CA}, {"cp": 4}), ({"thal": MISSING_THAL}, {"slope": 3})],
)
def test_a_bad_code_in_a_row_with_a_missing_code_is_refused(missing, bad):
    (name,) = bad
    with pytest.raises(ContractError, match=f"hd row 2: {name}"):
        from_hd(hd_text(hd_line(), hd_line(**missing, **bad)))
