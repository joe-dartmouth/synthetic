"""`bko.datasets.acs`: the ACS person file read as text, Folktables' filters and the frame records.

Every input is built in memory from `tests._acs_fixtures`; no test reads `data/`.
"""

from __future__ import annotations

import pytest

from bko.datasets.acs import HEADER, kept_rows, occupation_pairs, parse_acs
from bko.datasets.errors import ContractError
from tests._acs_fixtures import REAL_HEADER, REAL_ROW, row, text


# The reader


def test_the_header_must_be_the_published_names_in_order():
    assert tuple(REAL_HEADER.split(",")) == HEADER
    assert len(HEADER) == 286
    names = list(HEADER)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ContractError, match="header is not the published 286 names"):
        parse_acs(text(REAL_ROW, header=",".join(names)))
    with pytest.raises(ContractError, match="header"):
        parse_acs(text(header=REAL_HEADER + ",EXTRA"))


def test_a_row_with_the_wrong_field_count_names_its_line():
    with pytest.raises(ContractError, match="line 3: 287 fields, expected 286"):
        parse_acs(text(REAL_ROW, REAL_ROW + ",x"))
    with pytest.raises(ContractError, match="line 2: 285 fields, expected 286"):
        parse_acs(text(REAL_ROW.rpartition(",")[0]))


def test_an_unreadable_line_is_a_contract_error():
    with pytest.raises(ContractError, match="line 2"):
        parse_acs(text('P,"unterminated' + "\x00"))


def test_occp_keeps_its_leading_zeros():
    rows, _ = kept_rows(parse_acs(text(REAL_ROW)))
    assert rows.loc[1, "OCCP"] == "0420"
    assert rows.loc[1, "POBP"] == "025"
    assert rows.loc[1, "SCHL"] == "19"
    assert occupation_pairs(rows) == {"0420": "119151"}


def test_occp_to_socp_must_be_a_function():
    rows, _ = kept_rows(
        parse_acs(text(REAL_ROW, row(SOCP="119199"), row(OCCP="0010", SOCP="1110XX")))
    )
    with pytest.raises(ContractError, match="OCCP 0420 carries SOCP 119151, 119199, not one code"):
        occupation_pairs(rows)
    one, _ = kept_rows(parse_acs(text(REAL_ROW, REAL_ROW, row(OCCP="0010", SOCP="1110XX"))))
    assert occupation_pairs(one) == {"0010": "1110XX", "0420": "119151"}


@pytest.mark.parametrize(
    ("column", "dropped", "kept"),
    [
        ("AGEP", "16", "17"),
        ("PINCP", "100", "101"),
        ("WKHP", "0", "1"),
        ("PWGTP", "0", "1"),
    ],
)
def test_each_folktables_filter_drops_its_rows(column, dropped, kept):
    rows, counts = kept_rows(parse_acs(text(row(**{column: dropped}), row(**{column: kept}))))
    position = ("AGEP", "PINCP", "WKHP", "PWGTP").index(column)
    assert counts == (2, *([2] * position), *([1] * (4 - position)))
    assert list(rows.index) == [1]
    assert rows.loc[1, column] == kept


def test_a_blank_filter_value_is_kept_by_no_filter():
    rows, counts = kept_rows(parse_acs(text(row(PINCP=""), row(WKHP=""), REAL_ROW)))
    assert counts == (3, 3, 2, 1, 1)
    assert len(rows) == 1


def test_the_record_ids_count_the_kept_rows_in_file_order():
    rows, _ = kept_rows(
        parse_acs(text(row(AGEP="15"), row(AGEP="30"), row(PINCP="-50"), row(AGEP="40")))
    )
    assert list(rows.index) == [1, 2]
    assert list(rows["AGEP"]) == ["30", "40"]
    assert rows.index.name == "record_id"


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"AGEP": "x"}, "line 2: AGEP is 'x', not a whole number or blank"),
        ({"PINCP": "1.5"}, "PINCP is '1.5'"),
        ({"COW": ""}, "line 2: COW is '', not one of its codes"),
        ({"SEX": "3"}, "SEX is '3', not one of its codes"),
        ({"SCHL": "1"}, "SCHL is '1', not one of its codes"),
        ({"DEAR": "b"}, "DEAR is 'b', not one of its codes"),
        ({"OCCP": "420"}, "OCCP is '420'"),
        ({"POBP": ""}, "POBP is ''"),
        ({"SOCP": "11915"}, "SOCP is '11915', not six characters"),
        ({"SOCP": "11X151"}, "SOCP is '11X151'"),
    ],
)
def test_a_value_outside_the_dictionary_names_its_line(values, match):
    with pytest.raises(ContractError, match=match):
        kept_rows(parse_acs(text(row(**values))))
