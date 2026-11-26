"""`bko.datasets.acs`: the ACS person file read as text, Folktables' filters and the frame records.

Every input is built in memory from `tests._acs_fixtures`; no test reads `data/`.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.acs import (
    CODEBOOK,
    FEATURES,
    HEADER,
    MEMBER,
    frame_records,
    kept_rows,
    load_acs,
    occupation_pairs,
    parse_acs,
)
from bko.datasets.acs_frame import acs_frame
from bko.datasets.cells import LABEL, SEXES
from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records
from bko.datasets.sources import Source
from bko.ontology.acs_rows import ACS_FIELD_ROWS
from bko.ontology.schema import shown
from tests._acs_fixtures import REAL_HEADER, REAL_ROW, acs_archive, mixed_rows, row, text
from tests._datasets_fixtures import pin, plant


def _records(*rows: str) -> pd.DataFrame:
    return frame_records(*kept_rows(parse_acs(text(*rows))))


def _planted(tmp_path: Path, data: bytes) -> Source:
    source = pin(data, "acs-ma-2024")
    plant(tmp_path, source, data)
    return source


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


# The frame records


def test_the_label_is_income_strictly_above_50000():
    records = _records(row(PINCP="50000"), row(PINCP="50001"), row(PINCP="49999"))
    assert list(records[LABEL]) == [0, 1, 0]


def test_the_real_row_reads_as_its_values():
    records = _records(REAL_ROW)
    assert records.loc[1].to_dict() == {
        "age": 20,
        "sex": 1,
        "COW": 1,
        "SCHL": 19,
        "MAR": 5,
        "OCCP": 420,
        "POBP": 25,
        "WKHP": 40,
        "RAC1P": 1,
        "RELSHIPP": 38,
        "DEAR": 2,
        "DEYE": 2,
        "DREM": 2,
        "DPHY": 2,
        "DDRS": 2,
        "DOUT": 2,
        LABEL: 0,
    }
    assert (records.dtypes == "int64").all()


def test_sex_two_is_female_and_one_is_male():
    records = _records(row(SEX="2"), row(SEX="1"))
    assert [SEXES[code] for code in records["sex"]] == ["female", "male"]


def test_the_cohort_line_counts_each_filter(caplog):
    with caplog.at_level(logging.INFO, logger="bko.datasets.acs"):
        _records(row(AGEP="16"), row(PINCP="60000"), REAL_ROW)
    assert caplog.messages == [
        (
            "cohort dataset=acs rows=3 age_over_16=2 income_over_100=2 hours_over_0=2 "
            "weight_at_least_1=2 records=2 income_over_50000=1"
        )
    ]


def test_the_frame_holds_exactly_the_named_fields():
    spec = acs_frame(_records(*mixed_rows()))
    assert spec.features == (
        "age",
        "sex",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "WKHP",
        "RAC1P",
        "RELSHIPP",
        "DEAR",
        "DEYE",
        "DREM",
        "DPHY",
        "DDRS",
        "DOUT",
    )
    assert spec.features == FEATURES
    assert spec.numeric == ("age", "WKHP")
    assert spec.edges == (30, 45, 60)
    assert spec.codebook["OCCP"] == {10: "0010", 420: "0420", 1530: "1530", 6240: "6240"}
    assert spec.codebook["POBP"] == {25: "025"}
    assert spec.codebook["SCHL"][21] == "Bachelor's degree"
    assert spec.codebook["DEAR"] == {1: "Yes", 2: "No"}
    assert set(CODEBOOK) == {"COW", "SCHL", "MAR", "RAC1P", "RELSHIPP", *spec.features[10:]}
    assert [(r.field, r.low, r.high) for r in spec.ranges] == [("age", 25, 69), ("WKHP", 40, 40)]


def test_the_excluded_columns_never_enter_the_frame():
    records = _records(*mixed_rows())
    spec = acs_frame(records)
    for name in ("PINCP", "PWGTP", "SOCP", "ADJINC", "AGEP", "SEX", "RELP", "PUMA"):
        assert name not in spec.columns
        assert name not in records.columns
    check_records(records, spec)


def test_an_unused_column_never_changes_the_frame():
    plain = _records(REAL_ROW, row(OCCP="0010", SOCP="1110XX"))
    changed = _records(
        row(PUMA="00999", ADJINC="1000000", HINS1="1", INDP="9999"),
        row(OCCP="0010", SOCP="1110XX", NAICSP="999999", FOD1P="1100"),
    )
    pd.testing.assert_frame_equal(plain, changed)


def test_a_missing_value_shows_no_concept():
    dropped = row(AGEP="15", DEAR="", OCCP="", SOCP="", COW="")
    records = _records(dropped, REAL_ROW)
    assert len(records) == 1
    [record] = (
        {str(name): float(value) for name, value in kept.items()}
        for kept in records.to_dict("records")
    )
    assert shown(record, ACS_FIELD_ROWS) == {"OCCO:11915100"}
    with pytest.raises(ContractError, match="DEAR is '', not one of its codes"):
        _records(row(DEAR=""))


def test_the_frame_does_not_depend_on_file_order():
    rows = mixed_rows()
    shuffled = [*rows[7::-1], *rows[:7:-1]]
    first, second = acs_frame(_records(*rows)), acs_frame(_records(*shuffled))
    assert first.codebook == second.codebook
    assert first.ranges == second.ranges
    assert sorted(map(tuple, _records(*rows).to_numpy())) == sorted(
        map(tuple, _records(*shuffled).to_numpy())
    )


# The loader


def test_load_reads_the_member_of_a_verified_snapshot(tmp_path):
    source = _planted(tmp_path, acs_archive())
    loaded = load_acs(tmp_path, sources=(source,))
    pd.testing.assert_frame_equal(loaded, _records(*mixed_rows()))
    with pytest.raises(ValueError, match="one archive, not 2"):
        load_acs(tmp_path, sources=(source, source))


def test_load_refuses_an_occupation_code_with_two_soc_codes(tmp_path):
    source = _planted(tmp_path, acs_archive(REAL_ROW, row(SOCP="119199", PINCP="60000")))
    with pytest.raises(ContractError, match="OCCP 0420 carries SOCP 119151, 119199"):
        load_acs(tmp_path, sources=(source,))


def _corrupt_crc(data: bytes) -> bytes:
    start = data.index(MEMBER.encode()) + len(MEMBER)
    return data[:start] + bytes([data[start] ^ 0xFF]) + data[start + 1 :]


def _stored(body: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(MEMBER, body)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: b"not a zip archive", "BadZipFile"),
        (lambda: _corrupt_crc(acs_archive()), "BadZipFile"),
        (lambda: _stored(text(*mixed_rows()).encode("utf-16")), "UnicodeDecodeError"),
        (lambda: _stored(("\ufeff" + text(*mixed_rows())).encode()), "header"),
    ],
    ids=["not-a-zip", "bad-crc", "not-utf-8", "bom"],
)
def test_a_damaged_archive_is_a_contract_error(tmp_path, make, match):
    source = _planted(tmp_path, make())
    with pytest.raises(ContractError, match=match):
        load_acs(tmp_path, sources=(source,))


def test_a_member_larger_than_the_cap_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr("bko.datasets.archive.MEMBER_LIMIT", 100)
    source = _planted(tmp_path, acs_archive())
    with pytest.raises(ContractError, match=r"psam_p25.csv: \d+ bytes, over the 100-byte cap"):
        load_acs(tmp_path, sources=(source,))


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ((row(AGEP="16"),), "no records"),
        ((row(PINCP="50000"), REAL_ROW), "no income above 50000"),
    ],
    ids=["empty", "no-label-1"],
)
def test_load_refuses_a_cohort_it_cannot_score(tmp_path, rows, match):
    source = _planted(tmp_path, acs_archive(*rows))
    with pytest.raises(ContractError, match=match):
        load_acs(tmp_path, sources=(source,))
