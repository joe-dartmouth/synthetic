"""SUPPORT2's loader and frame over archives built in memory; no test reads `data/`."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records
from bko.datasets.sources import Source
from bko.datasets.states import Cutoffs, band_meanings
from bko.datasets.support2 import (
    AFTER_OUTCOME,
    COLUMNS,
    FIELDS,
    LEFT_OUT,
    MEMBER,
    WARNED,
    frame_records,
    load_support2,
    parse_support2,
)
from bko.datasets.support2_frame import support2_frame
from bko.ontology.schema import shown
from bko.ontology.support2_rows import SUPPORT2_FIELD_ROWS
from tests._datasets_fixtures import archive, pin, plant
from tests._support2_fixtures import REAL_HEADER, REAL_ROW, mixed_rows, row, support2_archive, text


def _records(*rows: str) -> pd.DataFrame:
    return frame_records(parse_support2(text(*rows)))


def _planted(tmp_path: Path, data: bytes) -> Source:
    source = pin(data, "support2")
    plant(tmp_path, source, data)
    return source


def test_the_header_must_be_the_archives_47_names_in_order() -> None:
    assert len(COLUMNS) == 47
    assert REAL_HEADER.split(",") == list(COLUMNS)
    swapped = REAL_HEADER.replace("age,death", "death,age")
    with pytest.raises(ContractError, match="header is not the archive's 47 names"):
        parse_support2(text(REAL_ROW, header=swapped))


@pytest.mark.parametrize(
    ("header", "rows", "line"),
    [
        (REAL_HEADER.replace("adlsc", "adlsc" + "x" * 200_000), (REAL_ROW,), 1),
        (REAL_HEADER, (row(1), row(2, sfdm2="x" * 200_000)), 3),
    ],
    ids=["header", "row"],
)
def test_an_unreadable_line_is_a_contract_error(
    header: str, rows: tuple[str, ...], line: int
) -> None:
    with pytest.raises(
        ContractError,
        match=rf"^support2\.csv line {line}: field larger than field limit \(131072\)$",
    ):
        parse_support2(text(*rows, header=header))


def test_a_quoted_field_keeps_its_comma() -> None:
    raw = parse_support2(text(row(1, sfdm2='"no(M2 and SIP pres), or not"'), row(2)))
    assert raw["sfdm2"].tolist() == ["no(M2 and SIP pres), or not", ""]
    assert raw["adlsc"].tolist() == ["7.0000000", "7.0000000"]


def test_crlf_line_endings_read_as_newlines() -> None:
    lines = text(row(1), row(2, hrt=" 91.0"))
    crlf = parse_support2(lines.replace("\n", "\r\n"))
    pd.testing.assert_frame_equal(crlf, parse_support2(lines))
    assert crlf["hrt"].tolist() == ["69.000000", "91.0"]


def test_a_row_holds_one_field_more_than_the_header() -> None:
    assert FIELDS == 48
    assert len(REAL_ROW.split(",")) == len(REAL_HEADER.split(",")) + 1


def test_a_row_with_the_wrong_field_count_names_its_line() -> None:
    short = REAL_ROW.rsplit(",", 1)[0]
    with pytest.raises(ContractError, match=r"support2\.csv line 3: 47 fields, expected 48"):
        parse_support2(text(row(1), short.replace("1,", "2,", 1)))


@pytest.mark.parametrize(
    ("number", "match"), [("x", "is not a whole number"), ("1", "record 1: listed more than once")]
)
def test_a_bad_or_repeated_row_number_is_refused(number: str, match: str) -> None:
    second = row(2).replace("2,", f"{number},", 1)
    with pytest.raises(ContractError, match=match):
        parse_support2(text(row(1), second))


def test_the_row_number_is_the_record_id() -> None:
    records = _records(row(7), row(3), row(12))
    assert records.index.name == "record_id"
    assert records.index.tolist() == [3, 7, 12]


def test_the_real_row_reads_as_its_values() -> None:
    [record] = _records(REAL_ROW).to_dict("records")
    assert record == {
        "age": 62,
        "sex": 1,
        "dzgroup": 6,
        "num.co": 0,
        "diabetes": 0,
        "dementia": 0,
        "ca": 2,
        "hday": 1,
        "scoma": 0,
        "race": 4,
        "income": 2,
        "edu": 2,
        "hrt": 1,
        "resp": 2,
        "temp": 2,
        "wblc": 2,
        "meanbp": 2,
        "crea": 2,
        "bili": 1,
        "pafi": 1,
        "sod": 3,
        "alb": 1,
        "ph": 3,
        "glucose": 0,
        "bun": 0,
        "urine": 0,
        "target": 0,
    }


def test_age_is_floored_to_whole_years() -> None:
    records = _records(row(1, age=" 62.99999"), row(2, age="18.0"), row(3, age=" 101.84796"))
    assert records["age"].tolist() == [62, 18, 101]


def test_the_label_is_hospdead() -> None:
    assert _records(row(1, hospdead="1"), row(2, hospdead="0"))["target"].tolist() == [1, 0]
    with pytest.raises(ContractError, match="record 1: hospdead is '2'"):
        _records(row(1, hospdead="2"))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dzgroup", "ARF", "dzgroup is 'ARF', not one of"),
        ("race", "unknown", "race is 'unknown'"),
        ("hrt", "fast", "hrt is 'fast', not a number or empty"),
        ("age", "", "age is '', not a number"),
        ("num_co", "1.5", "num.co is '1.5', not a whole number"),
        ("scoma", "9.5", "scoma is '9.5', not a whole number or empty"),
    ],
)
def test_a_value_outside_the_codebook_names_its_record(field: str, value: str, match: str) -> None:
    with pytest.raises(ContractError, match=f"record 2: {match}"):
        _records(row(2, **{field: value}), row(5))


def test_a_missing_scoma_drops_its_record_and_is_counted(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.support2"):
        records = _records(row(1), row(2, scoma=""), row(3, hospdead="1"))
    assert records.index.tolist() == [1, 3]
    assert caplog.messages == [
        "cohort dataset=support2 rows=3 missing_scoma=1 records=2 hospital_deaths=1"
    ]


def test_missing_race_and_income_are_code_0_not_recorded() -> None:
    records = _records(row(1, race="", income=""), row(2, race="hispanic", income=">$50k"))
    assert records[["race", "income"]].to_numpy().tolist() == [[0, 0], [5, 4]]
    frame = support2_frame(records)
    assert frame.codebook["race"][0] == frame.codebook["income"][0] == "not recorded"


def test_each_cited_field_is_coded_at_its_cutoffs() -> None:
    values = {
        "hrt": ("90.0", "91.0"),
        "resp": ("20", "21"),
        "meanbp": (" 70.0", " 69.0"),
        "pafi": ("400.0", "399.9"),
        "crea": (" 1.19995117", " 1.1"),
        "bili": (" 1.19995117", " 1.1"),
    }
    records = _records(
        row(1, **{k: v[0] for k, v in values.items()}),
        row(2, **{k: v[1] for k, v in values.items()}),
    )
    assert records[list(values)].to_dict("list") == {
        "hrt": [1, 2],
        "resp": [1, 2],
        "meanbp": [2, 1],
        "pafi": [2, 1],
        "crea": [2, 1],
        "bili": [2, 1],
    }
    temps = _records(*(row(n, temp=t) for n, t in enumerate(("35.9", "36", "38", "38.1"), 1)))
    assert temps["temp"].tolist() == [1, 2, 2, 3]
    wblc = _records(*(row(n, wblc=w) for n, w in enumerate(("3.9", "4", "12", "12.1"), 1)))
    assert wblc["wblc"].tolist() == [1, 2, 2, 3]


def test_a_missing_value_shows_no_concept() -> None:
    records = _records(row(1, crea="", hrt="", temp="", wblc=""), row(2))
    first, second = (
        {str(name): float(value) for name, value in record.items()}
        for record in records.to_dict("records")
    )
    assert [first[name] for name in ("crea", "hrt", "temp", "wblc")] == [0, 0, 0, 0]
    labels = {r.concept: r.label for r in SUPPORT2_FIELD_ROWS}
    lab = {"HP:0003259", "HP:0001649", "HP:0001945", "HP:0002045", "HP:0001974", "HP:0001882"}
    assert shown(first, SUPPORT2_FIELD_ROWS) & lab == set()
    present = shown(second, SUPPORT2_FIELD_ROWS) & lab
    assert {labels[c] for c in present} == {"Elevated circulating creatinine concentration"}


def test_the_excluded_columns_never_enter_the_frame() -> None:
    records = _records(*mixed_rows())
    frame = support2_frame(records)
    excluded = {*WARNED, *AFTER_OUTCOME, *LEFT_OUT, "hospdead"}
    assert excluded == {
        *("aps", "sps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday"),
        *("death", "slos", "d.time", "charges", "totcst", "totmcst", "avtisst", "sfdm2"),
        *("dzclass", "adlp", "adls", "adlsc", "hospdead"),
    }
    assert excluded.isdisjoint(records.columns)
    assert excluded.isdisjoint(frame.columns)


def test_the_frame_holds_exactly_the_named_fields() -> None:
    records = _records(*mixed_rows())
    frame = support2_frame(records)
    assert frame.columns == (
        *("age", "sex", "dzgroup", "num.co", "diabetes", "dementia", "ca", "hday", "scoma"),
        *("race", "income", "hrt", "resp", "temp", "wblc", "meanbp", "crea", "bili", "pafi"),
        *("edu", "sod", "alb", "ph", "glucose", "bun", "urine", "target"),
    )
    assert frame.numeric == ("age", "num.co", "hday", "scoma")
    assert frame.edges == (50, 65, 80)
    assert tuple(records.columns) == frame.columns
    assert {str(dtype) for dtype in records.dtypes} == {"int64"}
    check_records(records, frame)
    ranges = {bound.field: (bound.low, bound.high) for bound in frame.ranges}
    assert ranges["age"] == (40, 95)


def test_band_meanings_name_each_state() -> None:
    frame = support2_frame(_records(row(1)))
    assert dict(frame.codebook["temp"]) == {
        0: "not measured",
        1: "below 36",
        2: "at least 36 and at most 38",
        3: "above 38",
    }
    assert dict(frame.codebook["hrt"]) == {0: "not measured", 1: "at most 90", 2: "above 90"}
    assert band_meanings(Cutoffs()) == {0: "not measured", 1: "measured"}


def test_frame_bytes_do_not_depend_on_file_order() -> None:
    rows = mixed_rows()
    shuffled = [*rows[7::-1], *rows[:7:-1]]
    assert shuffled != rows
    assert _records(*shuffled).to_csv() == _records(*rows).to_csv()


def test_load_reads_the_member_of_a_verified_snapshot(tmp_path: Path) -> None:
    data = support2_archive()
    source = _planted(tmp_path, data)
    loaded = load_support2(tmp_path, sources=(source,))
    pd.testing.assert_frame_equal(loaded, _records(*mixed_rows()))
    with pytest.raises(ValueError, match="one archive, not 2"):
        load_support2(tmp_path, sources=(source, source))


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
        (lambda: _corrupt_crc(support2_archive()), "BadZipFile"),
        (lambda: _stored(text(*mixed_rows()).encode("utf-16")), "UnicodeDecodeError"),
        (lambda: archive(chr(0xFEFF) + text(*mixed_rows()), member=MEMBER), "header"),
    ],
    ids=["not-a-zip", "bad-crc", "not-utf-8", "bom"],
)
def test_a_damaged_archive_is_a_contract_error(tmp_path: Path, make, match: str) -> None:
    source = _planted(tmp_path, make())
    with pytest.raises(ContractError, match=match):
        load_support2(tmp_path, sources=(source,))


def test_a_member_larger_than_the_cap_is_refused_before_it_is_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("bko.datasets.archive.MEMBER_LIMIT", 100)
    source = _planted(tmp_path, support2_archive())
    with pytest.raises(ContractError, match=r"support2.csv: \d+ bytes, over the 100-byte cap"):
        load_support2(tmp_path, sources=(source,))


@pytest.mark.parametrize(
    ("rows", "match"),
    [((row(1, scoma=""),), "no records"), ((row(1), row(2)), "no in-hospital death")],
    ids=["empty", "no-death"],
)
def test_load_refuses_a_cohort_it_cannot_score(tmp_path: Path, rows, match: str) -> None:
    source = _planted(tmp_path, support2_archive(*rows))
    with pytest.raises(ContractError, match=match):
        load_support2(tmp_path, sources=(source,))
