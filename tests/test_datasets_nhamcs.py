"""NHAMCS's loader over archives built in memory; no test reads `data/`."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records, digest
from bko.datasets.icd10_tokens import token_number
from bko.datasets.nhamcs import (
    DIAGNOSES,
    FLAGS,
    MEMBER,
    POSITIONS,
    WIDTH,
    frame_records,
    load_nhamcs,
    parse_nhamcs,
)
from bko.datasets.nhamcs_frame import nhamcs_frame
from bko.datasets.sources import Source
from bko.ontology.icd10cm_codes import ICD10CM_CODES
from bko.ontology.nhamcs_rows import DIAGNOSIS_ROWS, FLAG_ROWS, NHAMCS_FIELD_ROWS
from bko.ontology.schema import Codes, shown
from tests._datasets_fixtures import archive, pin, plant
from tests._nhamcs_fixtures import (
    BLANK_CHRONIC,
    LATE_TOKENS,
    MIXED_DIGEST,
    REAL_ADMITTED,
    REAL_LINE,
    chronic,
    line,
    mixed,
    nhamcs_archive,
    text,
)


def _records(*lines: str) -> pd.DataFrame:
    return frame_records(parse_nhamcs(text(*lines)))


def _planted(tmp_path: Path, data: bytes) -> Source:
    source = pin(data, "nhamcs-ed-2022")
    plant(tmp_path, source, data)
    return source


def test_the_positions_neither_overlap_nor_run_past_the_record() -> None:
    spans = sorted(POSITIONS.values())
    assert all(first <= last for first, last in spans)
    assert all(last < first for (_, last), (first, _) in pairwise(spans))
    assert spans[0][0] >= 1
    assert spans[-1][1] <= WIDTH == 2382
    assert dict(POSITIONS) == {
        "AGE": (16, 18),
        "SEX": (25, 25),
        "DIAG1": (122, 125),
        "DIAG2": (126, 129),
        "DIAG3": (130, 133),
        "DIAG4": (134, 137),
        "DIAG5": (138, 141),
        **{name: (column, column) for name, column in zip(FLAGS, range(152, 174), strict=True)},
        "NOCHRON": (174, 174),
        "ADMITHOS": (499, 499),
        "HOSPCODE": (544, 546),
        "PATCODE": (547, 549),
    }
    assert FLAGS[0] == "ETOHAB"
    assert FLAGS[-1] == "SUBSTAB"


def test_a_record_parses_to_its_literal_fields() -> None:
    raw = parse_nhamcs(text(REAL_LINE))
    assert raw.index.tolist() == [34001]
    [record] = raw.to_dict("records")
    assert {name: record[name] for name in ("AGE", "SEX", *DIAGNOSES, "NOCHRON", "ADMITHOS")} == {
        "AGE": "023",
        "SEX": "1",
        "DIAG1": "T833",
        "DIAG2": "-9  ",
        "DIAG3": "-9  ",
        "DIAG4": "-9  ",
        "DIAG5": "-9  ",
        "NOCHRON": "1",
        "ADMITHOS": "0",
    }
    [coded] = _records(REAL_LINE).to_dict("records")
    assert coded == {
        "age": 23,
        "sex": 0,
        "DIAG1": 30 * 37**3 + 9 * 37**2 + 4 * 37 + 4,
        "DIAG2": 0,
        "DIAG3": 0,
        "DIAG4": 0,
        "DIAG5": 0,
        **dict.fromkeys(FLAGS, 0),
        "target": 0,
    }


def test_an_admitted_record_with_chronic_items_parses_to_its_literal_fields() -> None:
    raw = parse_nhamcs(text(REAL_ADMITTED))
    assert raw.index.tolist() == [34012]
    [record] = raw.to_dict("records")
    assert {name: record[name] for name in ("AGE", "SEX", *DIAGNOSES, "NOCHRON", "ADMITHOS")} == {
        "AGE": "065",
        "SEX": "1",
        "DIAG1": "K859",
        "DIAG2": "R109",
        "DIAG3": "-9  ",
        "DIAG4": "-9  ",
        "DIAG5": "-9  ",
        "NOCHRON": "0",
        "ADMITHOS": "1",
    }
    ticked = {"ETOHAB", "DEPRN", "DIABTYP1", "HYPLIPID", "OBESITY"}
    assert {name for name in FLAGS if record[name] == "1"} == ticked
    [coded] = _records(REAL_ADMITTED).to_dict("records")
    assert coded == {
        "age": 65,
        "sex": 0,
        "DIAG1": 21 * 37**3 + 9 * 37**2 + 6 * 37 + 10,
        "DIAG2": 28 * 37**3 + 2 * 37**2 + 1 * 37 + 10,
        "DIAG3": 0,
        "DIAG4": 0,
        "DIAG5": 0,
        **{name: int(name in ticked) for name in FLAGS},
        "target": 1,
    }


def test_a_line_of_the_wrong_length_names_its_line() -> None:
    with pytest.raises(ContractError, match=r"^ed2022 line 2: 2381 characters, expected 2382$"):
        parse_nhamcs(text(line(1), line(2)[:-1], line(3)))
    with pytest.raises(ContractError, match=r"^ed2022 line 1: 4765 characters, expected 2382$"):
        parse_nhamcs(f"{line(1)}\n{line(2)}\r\n")


def test_a_text_without_its_last_crlf_is_refused() -> None:
    with pytest.raises(ContractError, match=r"^ed2022: the last line has no CRLF$"):
        parse_nhamcs(text(line(1)).removesuffix("\r\n"))
    assert parse_nhamcs("").empty


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda s: s[:543] + " 34" + s[546:], r"HOSPCODE ' 34' or PATCODE '001' is not three"),
        (lambda s: s, r"record 34001: listed more than once"),
    ],
    ids=["not-digits", "repeated"],
)
def test_a_bad_or_repeated_visit_id_is_refused(edit, match: str) -> None:
    with pytest.raises(ContractError, match=match):
        parse_nhamcs(text(line(1), edit(line(1))))


def test_the_id_is_the_hospital_and_the_visit() -> None:
    records = _records(line(7, HOSPCODE="188"), line(3, HOSPCODE="002"), line(173, HOSPCODE="002"))
    assert records.index.name == "record_id"
    assert records.index.tolist() == [2003, 2173, 188007]


def test_admithos_is_the_label() -> None:
    records = _records(line(1, ADMITHOS="1"), line(2, ADMITHOS="0"))
    assert records["target"].tolist() == [1, 0]
    with pytest.raises(ContractError, match=r"record 34001: ADMITHOS is '2', not 0 or 1"):
        _records(line(1, ADMITHOS="2"))


def test_age_is_as_published_and_sex_is_recoded() -> None:
    records = _records(line(1, AGE="000", SEX="2"), line(2, AGE="094", SEX="1"))
    assert records[["age", "sex"]].to_numpy().tolist() == [[0, 1], [94, 0]]


@pytest.mark.parametrize(
    ("items", "match"),
    [
        ({"AGE": "095"}, "AGE is '095', not 000 to 094"),
        ({"AGE": " 23"}, "AGE is ' 23', not 000 to 094"),
        ({"SEX": "0"}, "SEX is '0', not 1 or 2"),
        ({"DIAG3": "ZZZ1"}, "DIAG3 is 'ZZZ1', not a diagnosis token"),
        ({"DIAG5": "-9-9"}, "DIAG5 is '-9-9', not a diagnosis token"),
        ({"CHF": "2", "NOCHRON": "0"}, "CHF is '2', not 0 or 1"),
        ({"NOCHRON": "3"}, "NOCHRON is '3', not 0, 1 or 2"),
        ({**chronic(), "NOCHRON": "0"}, "NOCHRON is '0', at odds with the flags"),
        ({**chronic("HTN"), "NOCHRON": "1"}, "NOCHRON is '1', at odds with the flags"),
        ({**chronic("HTN"), "NOCHRON": "2"}, "NOCHRON is '2', at odds with the flags"),
    ],
)
def test_a_value_outside_its_codes_names_its_record(items: dict[str, str], match: str) -> None:
    with pytest.raises(ContractError, match=f"record 34002: {re.escape(match)}"):
        _records(line(5), line(2, **items))


def test_each_diagnosis_column_holds_its_token_number() -> None:
    records = _records(line(1, DIAG1="A00-", DIAG2="A000", DIAG3="ZZZ5", DIAG4="I509"))
    assert records[list(DIAGNOSES)].to_numpy().tolist() == [
        [
            11 * 37**3 + 1 * 37**2 + 1 * 37,
            11 * 37**3 + 1 * 37**2 + 1 * 37 + 1,
            36 * 37**3 + 36 * 37**2 + 36 * 37 + 6,
            19 * 37**3 + 6 * 37**2 + 1 * 37 + 10,
            0,
        ]
    ]


def test_a_blank_chronic_item_codes_every_flag_not_recorded_and_is_counted(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.nhamcs"):
        records = _records(
            line(1, **BLANK_CHRONIC), line(2, ADMITHOS="1", **chronic("CHF", "HTN")), line(3)
        )
    assert records.loc[[34001], list(FLAGS)].to_numpy().tolist()[0] == [2] * len(FLAGS)
    assert records.loc[[34002], ["CHF", "HTN", "ASTHMA"]].to_numpy().tolist()[0] == [1, 1, 0]
    assert records.loc[[34003], list(FLAGS)].to_numpy().tolist()[0] == [0] * len(FLAGS)
    assert caplog.messages == ["cohort dataset=nhamcs rows=3 records=3 admitted=1 chronic_blank=1"]


def test_frame_bytes_do_not_depend_on_file_order() -> None:
    lines = mixed()
    shuffled = [*lines[7::-1], *lines[:7:-1]]
    assert shuffled != lines
    assert digest(_records(*shuffled)) == digest(_records(*lines)) == MIXED_DIGEST


def test_load_reads_the_member_of_a_verified_snapshot(tmp_path: Path) -> None:
    source = _planted(tmp_path, nhamcs_archive())
    loaded = load_nhamcs(tmp_path, sources=(source,))
    pd.testing.assert_frame_equal(loaded, _records(*mixed()))
    with pytest.raises(ValueError, match="one archive, not 2"):
        load_nhamcs(tmp_path, sources=(source, source))


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
        (lambda: _corrupt_crc(nhamcs_archive()), "BadZipFile"),
        (lambda: _stored(text(*mixed()).encode("utf-16")), "UnicodeDecodeError"),
        (lambda: archive(text(*mixed()), member="ed2021"), "the archive has no such member"),
        (
            lambda: archive(chr(0xFEFF) + text(*mixed()), member=MEMBER),
            r"^ed2022 line 1: 2383 characters, expected 2382$",
        ),
    ],
    ids=["not-a-zip", "bad-crc", "not-utf-8", "no-member", "bom"],
)
def test_a_damaged_archive_is_a_contract_error(tmp_path: Path, make, match: str) -> None:
    source = _planted(tmp_path, make())
    with pytest.raises(ContractError, match=match):
        load_nhamcs(tmp_path, sources=(source,))


def test_a_member_larger_than_the_cap_is_refused_before_it_is_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("bko.datasets.archive.MEMBER_LIMIT", 100)
    source = _planted(tmp_path, nhamcs_archive())
    with pytest.raises(ContractError, match=r"ed2022: \d+ bytes, over the 100-byte cap"):
        load_nhamcs(tmp_path, sources=(source,))


@pytest.mark.parametrize(
    ("lines", "match"),
    [((), "holds no records"), ((line(1), line(2)), "holds no admission")],
    ids=["empty", "no-admission"],
)
def test_load_refuses_records_it_cannot_score(tmp_path: Path, lines, match: str) -> None:
    data = archive(text(*lines), member=MEMBER)
    source = _planted(tmp_path, data)
    with pytest.raises(ContractError, match=match):
        load_nhamcs(tmp_path, sources=(source,))


def test_the_disposition_and_stay_fields_never_enter() -> None:
    read = {name for name, (first, last) in POSITIONS.items() if last >= 487}
    assert read == {"ADMITHOS", "HOSPCODE", "PATCODE"}
    records = _records(*mixed())
    excluded = {
        *("ADMITHOS", "ADMIT", "LOS", "BOARDED", "PATWT", "EDWT"),
        *("NOCHRON", "TOTCHRON", "HOSPCODE", "PATCODE"),
    }
    assert excluded.isdisjoint(records.columns)
    assert excluded.isdisjoint(nhamcs_frame(records).columns)


def test_the_frame_holds_exactly_the_named_fields() -> None:
    records = _records(*mixed())
    frame = nhamcs_frame(records)
    assert frame.columns == (
        *("age", "sex", "DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"),
        *("ETOHAB", "ALZHD", "ASTHMA", "CANCER", "CEBVD", "CKD", "COPD", "CHF", "CAD", "DEPRN"),
        *("DIABTYP1", "DIABTYP2", "DIABTYP0", "ESRD", "HPE", "EDHIV", "HYPLIPID", "HTN"),
        *("OBESITY", "OSA", "OSTPRSIS", "SUBSTAB", "target"),
    )
    assert frame.numeric == ("age",)
    assert frame.edges == (18, 45, 65)
    assert tuple(records.columns) == frame.columns
    assert {str(dtype) for dtype in records.dtypes} == {"int64"}
    check_records(records, frame)
    assert {bound.field: (bound.low, bound.high) for bound in frame.ranges} == {"age": (5, 94)}
    assert dict(frame.codebook["HTN"]) == {0: "no", 1: "yes", 2: "not recorded"}
    book = {
        0: "-9 (none recorded)",
        token_number("A00-"): "A00-",
        token_number("A000"): "A000",
        token_number("E119"): "E119",
        token_number("I509"): "I509",
        token_number("J45-"): "J45-",
        token_number("N179"): "N179",
        token_number("R079"): "R079",
        token_number("ZZZ0"): "ZZZ0",
    }
    for column in DIAGNOSES:
        assert dict(frame.codebook[column]) == book, column
    for column, token in LATE_TOKENS.items():
        assert {name for name in DIAGNOSES if (records[name] == token_number(token)).any()} == {
            column
        }
        assert all(token_number(token) in frame.codebook[name] for name in DIAGNOSES), token
    assert frame.label_meanings == {
        0: "not admitted to this hospital",
        1: "admitted to this hospital",
    }


def test_a_visit_with_the_chronic_item_blank_shows_no_flag_concept() -> None:
    records = _records(
        line(1, **BLANK_CHRONIC),
        line(2, **chronic(*FLAGS)),
        line(3, **chronic()),
    )
    blank, every, none = (
        {str(name): float(value) for name, value in record.items()}
        for record in records.to_dict("records")
    )
    assert shown(blank, FLAG_ROWS) == shown(none, FLAG_ROWS) == set()
    assert shown(every, FLAG_ROWS) == {row.concept for row in FLAG_ROWS}


def test_a_missing_value_shows_no_concept() -> None:
    record = {
        str(name): float(value)
        for name, value in _records(line(1, **BLANK_CHRONIC)).iloc[0].items()
    }
    assert record["DIAG2"] == 0
    assert all(record[name] == 2 for name in FLAGS)
    concepts = shown(record, NHAMCS_FIELD_ROWS)
    assert not concepts & {row.concept for row in FLAG_ROWS}
    diagnosis_codes = [
        clause.codes
        for row in DIAGNOSIS_ROWS
        for clause in row.when
        if isinstance(clause, Codes) and clause.field in DIAGNOSES
    ]
    assert len(diagnosis_codes) == len(DIAGNOSIS_ROWS) > 0
    assert all(0 not in codes for codes in diagnosis_codes)


def test_every_field_row_reads_a_frame_column_within_its_codebook() -> None:
    lines = [line(n, DIAG1=row.token) for n, row in enumerate(ICD10CM_CODES, 1)]
    frame = nhamcs_frame(_records(*lines))
    clauses = [clause for row in NHAMCS_FIELD_ROWS for clause in row.when]
    assert set(DIAGNOSES) <= {clause.field for clause in clauses if isinstance(clause, Codes)}
    for clause in clauses:
        assert isinstance(clause, Codes)
        assert clause.field in frame.columns
        assert clause.codes <= frame.codebook[clause.field].keys(), clause
