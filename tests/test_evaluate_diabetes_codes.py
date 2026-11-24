"""The `diabetes-codes` dataset: Diabetes 130's cohort with each diagnosis code kept.

No test reaches the API or reads `data/`: the archive is the toy cohort `toy_archive` builds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Final

import pandas as pd
import pytest

from bko.datasets.diabetes import DIAGNOSES, load_diabetes
from bko.datasets.diagnoses import code_number, code_text
from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, digest, field_rows, frame, load
from bko.ontology.code_frames import (
    DIABETES_CODES_FIELD_ROWS,
    DIABETES_CODES_NAME,
    diabetes_codes_frame,
)
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    LAB_FIELD_ROWS,
)
from bko.ontology.icd9 import ICD9_MAPPINGS
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.schema import Codes
from tests._diabetes_fixtures import pin, toy_archive
from tests._ontology_fixtures import RoutingOpener


TOY_TOKENS: Final = (
    *("250.41", "401", "410", "428.0", "434.91", "486", "491.21", "496", "585", "707.1"),
    *("8", "V45", "?", "276", "E888"),
)
"""Every diagnosis token `toy_archive` writes, listed here rather than read from the fixture."""

ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context() -> Context:
    return Context(open_url=RoutingOpener({PIN.url: ARCHIVE}), diabetes=PIN)


def test_the_diabetes_codes_frame_keeps_each_diagnosis_code(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.DIABETES_CODES, tmp_path, _context())
    records = data.records
    assert tuple(records.columns) == data.frame.columns
    assert data.frame.name == DIABETES_CODES_NAME == "diabetes-codes"
    assert set(DIAGNOSES) <= set(data.frame.categorical)
    assert set(DIABETES_INDICATORS).isdisjoint(records.columns)
    raw = {code_text(int(n)) for name in DIAGNOSES for n in records[name]}
    assert raw == set(TOY_TOKENS)
    for name in DIAGNOSES:
        book = data.frame.codebook[name]
        assert book == {
            code_number(token): "? (none recorded)" if token == "?" else token
            for token in TOY_TOKENS
        }
        assert next(iter(book.items())) == (0, "? (none recorded)")
        assert book[code_number("428.0")] == "428.0"
    assert data.frame.features[-3:] == DIAGNOSES
    assert {name: data.frame.descriptions[name] for name in DIAGNOSES} == {
        "diag_1": "ICD-9 code of the encounter's primary diagnosis (? = none recorded)",
        "diag_2": "ICD-9 code of the encounter's secondary diagnosis (? = none recorded)",
        "diag_3": (
            "ICD-9 code of the encounter's additional secondary diagnosis (? = none recorded)"
        ),
    }
    assert (data.field_rows, data.name) == (DIABETES_CODES_FIELD_ROWS, PIN.name)
    line = (
        f"data dataset=diabetes-codes frame=diabetes-codes records={len(records)} "
        f"digest={digest(records)} agreement=none"
    )
    assert line in caplog.messages


def test_diabetes_codes_has_the_cohort_and_label_of_diabetes(tmp_path: Path) -> None:
    codes = load(Dataset.DIABETES_CODES, tmp_path, _context())
    plain = load(Dataset.DIABETES, tmp_path, _context())
    shared = [name for name in plain.records.columns if name not in DIABETES_INDICATORS]
    pd.testing.assert_frame_equal(codes.records[shared], plain.records[shared])
    assert codes.frame.label == plain.frame.label
    assert codes.frame.edges == DIABETES_FRAME.edges
    assert codes.frame.ranges == DIABETES_FRAME.ranges
    direct = load_diabetes(tmp_path, indicators={}, diagnoses=True, source=PIN)
    assert direct.equals(codes.records)
    assert diabetes_codes_frame(direct).codebook == codes.frame.codebook


def test_field_rows_of_diabetes_codes_need_no_data() -> None:
    context = Context(open_url=_refusing_opener)
    assert field_rows(Dataset.DIABETES_CODES, context) == DIABETES_CODES_FIELD_ROWS


def test_the_frame_of_diabetes_codes_needs_its_records() -> None:
    with pytest.raises(ValueError, match="diabetes-codes"):
        frame(Dataset.DIABETES_CODES)


def test_each_diagnosis_columns_rows_partition_its_mapped_codes() -> None:
    mapped = {code_number(row.code) for row in ICD9_CODES}
    code_rows = DIABETES_CODES_FIELD_ROWS[: len(DIABETES_CODES_FIELD_ROWS) - len(LAB_FIELD_ROWS)]
    assert DIABETES_CODES_FIELD_ROWS[len(code_rows) :] == LAB_FIELD_ROWS
    assert len(code_rows) == len(DIAGNOSES) * len(ICD9_CODES)
    for name in DIAGNOSES:
        held: list[int] = []
        for row in code_rows:
            [clause] = row.when
            assert isinstance(clause, Codes)
            if clause.field == name:
                held.extend(clause.codes)
        assert sorted(held) == sorted(mapped), name
    concepts = {code_number(row.code): row.concept for row in ICD9_CODES}
    for row in code_rows:
        [clause] = row.when
        assert isinstance(clause, Codes)
        [number] = clause.codes
        assert row.concept == concepts[number]


def test_the_diabetes_field_rows_are_unchanged() -> None:
    icd9 = DIABETES_FIELD_ROWS[: len(ICD9_MAPPINGS)]
    assert (*icd9, *LAB_FIELD_ROWS) == DIABETES_FIELD_ROWS
    assert [row.concept for row in icd9] == [row.concept for row in ICD9_MAPPINGS]
    assert [row.when for row in icd9] == [
        (Codes(name, frozenset({1})),) for name in DIABETES_INDICATORS
    ]


def test_records_outside_the_codes_frames_codebook_are_refused(tmp_path: Path) -> None:
    data = load(Dataset.DIABETES_CODES, tmp_path, _context())
    check_records(data.records, data.frame)
    stray = data.records.copy()
    stray.loc[stray.index[0], "diag_2"] = code_number("999.99")
    with pytest.raises(ContractError, match="diag_2"):
        check_records(stray, data.frame)
