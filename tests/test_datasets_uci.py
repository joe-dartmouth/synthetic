"""Parsing the Cleveland file into the evaluation frame, and loading it from a verified snapshot."""

from __future__ import annotations

import logging

import pytest

from bko.datasets.errors import ContractError, UnavailableError
from bko.datasets.uci import (
    CATEGORICAL,
    CODEBOOK,
    COLUMNS,
    FEATURES,
    FRAME_COLUMNS,
    FRAME_DTYPES,
    LABEL,
    NUMERIC,
    complete_records,
    load_uci,
    parse_cleveland,
)
from tests._datasets_fixtures import REAL_LAYOUT, archive, cleveland_line, pin, plant, with_field


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_the_real_layout_parses_to_fourteen_named_fields():
    frame = parse_cleveland(REAL_LAYOUT + "\n")
    assert list(frame.columns) == list(COLUMNS)
    assert frame.loc[1, "oldpeak"] == pytest.approx(2.3)
    assert frame.loc[1, "thal"] == pytest.approx(6.0)


@pytest.mark.parametrize("field", ["ca", "thal"])
def test_a_question_mark_in_ca_or_thal_reads_as_missing(field):
    frame = parse_cleveland(with_field(cleveland_line(), field, "?"))
    assert frame[field].isna().all()


@pytest.mark.parametrize("field", ["age", "sex", "num"])
def test_a_missing_age_sex_or_num_is_refused(field):
    with pytest.raises(ContractError, match=f"line 1: {field} is missing"):
        parse_cleveland(with_field(REAL_LAYOUT, field, "?"))


def test_a_wrong_field_count_names_its_line():
    text = cleveland_line() + "\n" + ",".join(REAL_LAYOUT.split(",")[:13]) + "\n"
    with pytest.raises(ContractError, match="line 2: 13 fields, expected 14"):
        parse_cleveland(text)


def test_a_non_number_is_refused_naming_the_field():
    with pytest.raises(ContractError, match="line 1: thalach is 'abc'"):
        parse_cleveland(cleveland_line(thalach="abc"))


@pytest.mark.parametrize("token", ["1.5", "nan", "inf"])
def test_a_fractional_or_non_finite_code_is_refused(token):
    with pytest.raises(ContractError, match="line 1: cp"):
        parse_cleveland(cleveland_line(cp=token))


@pytest.mark.parametrize(
    ("field", "token"), [("cp", "5.0"), ("thal", "4.0"), ("ca", "4.0"), ("num", "5")]
)
def test_a_code_outside_its_codebook_is_refused(field, token):
    with pytest.raises(ContractError, match=f"line 1: {field} is"):
        parse_cleveland(with_field(cleveland_line(), field, token))


def test_an_empty_member_is_refused():
    with pytest.raises(ContractError, match="no records"):
        parse_cleveland("\n")


def test_crlf_and_trailing_blank_lines_parse_like_lf():
    lf = parse_cleveland(cleveland_line() + "\n" + cleveland_line(age=41) + "\n")
    crlf = parse_cleveland(cleveland_line() + "\r\n" + cleveland_line(age=41) + "\r\n\r\n")
    assert lf.equals(crlf)


def test_record_id_counts_blank_lines():
    """A blank or whitespace-only line keeps its number, so later ids stay line numbers."""
    text = "\n".join([cleveland_line(), "", "   ", cleveland_line(age=41)])
    assert list(parse_cleveland(text).index) == [1, 4]


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_a_non_finite_oldpeak_is_refused(token):
    with pytest.raises(ContractError, match=r"line 1: oldpeak is .*, not a finite number"):
        parse_cleveland(cleveland_line(oldpeak=token))


def test_padded_fields_parse_like_bare_ones():
    padded = parse_cleveland(with_field(with_field(cleveland_line(), "cp", " 1.0 "), "ca", " ? "))
    assert padded.loc[1, "cp"] == pytest.approx(1.0)
    assert padded["ca"].isna().all()


def test_record_id_is_the_line_number():
    frame = parse_cleveland("\n".join(cleveland_line(age=40 + i) for i in range(3)))
    assert list(frame.index) == [1, 2, 3]
    assert frame.index.name == "record_id"


@pytest.mark.parametrize(("num", "target"), [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)])
def test_num_above_zero_is_disease(num, target):
    frame = complete_records(parse_cleveland(cleveland_line(num=num)))
    assert frame[LABEL].tolist() == [target]


def test_num_is_not_a_column_of_the_frame():
    frame = complete_records(parse_cleveland(cleveland_line()))
    assert list(frame.columns) == [*FEATURES, LABEL]
    assert "num" not in frame.columns


def test_incomplete_records_are_dropped_and_logged(caplog):
    text = "\n".join([cleveland_line(), cleveland_line(ca="?"), cleveland_line(thal="?")])
    with caplog.at_level(logging.WARNING):
        frame = complete_records(parse_cleveland(text))
    assert list(frame.index) == [1]
    assert "dropped 2 records with a missing value, record_ids=2,3" in caplog.text


def test_a_member_with_no_complete_record_is_refused():
    with pytest.raises(ContractError, match="no complete records"):
        complete_records(parse_cleveland(cleveland_line(ca="?")))


def test_features_are_integers_except_oldpeak():
    frame = complete_records(parse_cleveland(cleveland_line()))
    assert frame["oldpeak"].dtype == "float64"
    assert {str(frame[name].dtype) for name in FEATURES if name != "oldpeak"} == {"int64"}
    assert frame[LABEL].dtype == "int64"


def test_the_codebook_covers_exactly_the_coded_features():
    assert set(CODEBOOK) == set(CATEGORICAL)
    assert set(NUMERIC).isdisjoint(CATEGORICAL)
    assert set(NUMERIC) | set(CATEGORICAL) == set(FEATURES)


def test_load_uci_reads_the_member_of_a_verified_snapshot(tmp_path):
    data = archive("\n".join(cleveland_line(age=40 + i) for i in range(4)) + "\n")
    source = pin(data)
    plant(tmp_path, source, data)
    assert len(load_uci(tmp_path, source=source)) == 4


def test_load_uci_refuses_an_archive_without_the_member(tmp_path):
    data = archive(cleveland_line(), member="processed.va.data")
    source = pin(data)
    plant(tmp_path, source, data)
    with pytest.raises(ContractError, match=r"cannot read processed\.cleveland\.data"):
        load_uci(tmp_path, source=source)


@pytest.mark.parametrize(
    "data",
    [b"not a zip archive", archive(cleveland_line() + "\u00e9")],
    ids=["not-a-zip", "not-ascii"],
)
def test_load_uci_refuses_an_unreadable_member(tmp_path, data):
    source = pin(data)
    plant(tmp_path, source, data)
    pattern = r"cannot read processed\.cleveland\.data: (BadZipFile|UnicodeDecodeError): "
    with pytest.raises(ContractError, match=pattern):
        load_uci(tmp_path, source=source)


def test_load_uci_refuses_a_snapshot_that_fails_its_pin(tmp_path):
    data = archive(cleveland_line())
    source = pin(data)
    plant(tmp_path, source, data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ContractError, match="pinned"):
        load_uci(tmp_path, source=source)


def test_load_uci_never_fetches(tmp_path):
    with pytest.raises(UnavailableError, match="no snapshot"):
        load_uci(tmp_path, source=pin(archive(cleveland_line())))


def test_the_frame_constants_describe_the_evaluation_frame():
    frame = complete_records(parse_cleveland(cleveland_line() + "\n"))
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert frame.dtypes.astype(str).to_dict() == dict(FRAME_DTYPES)
