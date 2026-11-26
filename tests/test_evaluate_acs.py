"""The `acs` dataset through `bko.evaluate.data`, read from a catalog entry with fixture pins, and
ACS's field rows.

No test reaches the API or reads `data/`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import BinaryIO

import pytest

from bko.datasets.sources import ACS_MA_2024, HPO, OCCO
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, dataset_releases, digest, field_rows, frame, load
from bko.ontology.acs_rows import ACS_FIELD_ROWS, DISABILITY_ROWS, OCCUPATION_CODES
from bko.ontology.catalog import CATALOG
from bko.ontology.occupation_codes_high import OCCUPATION_CODES as HIGH
from bko.ontology.occupation_codes_low import OCCUPATION_CODES as LOW
from bko.ontology.schema import Codes
from tests._acs_fixtures import acs_archive
from tests._datasets_fixtures import pin
from tests._ontology_fixtures import RoutingOpener


ARCHIVE = acs_archive()
PIN = pin(ARCHIVE, "acs-ma-2024")


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context(opener=None, **entry) -> Context:
    catalogued = dataclasses.replace(CATALOG["acs"], sources=(PIN,), **entry)
    return Context(
        open_url=RoutingOpener({PIN.url: ARCHIVE}) if opener is None else opener,
        catalog={"acs": catalogued},
    )


def test_the_acs_dataset_loads_and_logs_its_data_line(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.ACS, tmp_path, _context())
    assert Dataset.ACS.value == "acs"
    assert data.frame.name == "acs"
    assert tuple(data.records.columns) == data.frame.columns
    assert len(data.records) == 12
    assert (data.field_rows, data.name, data.agreement, data.held) == (
        ACS_FIELD_ROWS,
        "acs",
        None,
        None,
    )
    line = f"data dataset=acs frame=acs records=12 digest={digest(data.records)} agreement=none"
    assert line in caplog.messages


def test_field_rows_of_acs_need_no_data() -> None:
    assert field_rows(Dataset.ACS, _context(_refusing_opener)) == ACS_FIELD_ROWS


def test_acs_reads_its_own_releases() -> None:
    context = _context(_refusing_opener)
    assert dict(dataset_releases(Dataset.ACS, context)) == {"HP": HPO, "OCCO": OCCO}
    assert dataset_releases(Dataset.SUPPORT2, Context()) is Context().releases
    assert CATALOG["acs"].sources == (ACS_MA_2024,)


def test_the_frame_of_acs_needs_its_records() -> None:
    with pytest.raises(ValueError, match="acs"):
        frame(Dataset.ACS)


def test_every_occp_code_is_in_exactly_one_row() -> None:
    assert (*LOW, *HIGH) == OCCUPATION_CODES
    occupation_rows = ACS_FIELD_ROWS[len(DISABILITY_ROWS) :]
    assert ACS_FIELD_ROWS[: len(DISABILITY_ROWS)] == DISABILITY_ROWS
    held: list[int] = []
    concepts = {int(row.code): row.concept for row in OCCUPATION_CODES}
    for row in occupation_rows:
        [clause] = row.when
        assert isinstance(clause, Codes)
        assert clause.field == "OCCP"
        [code] = clause.codes
        held.append(code)
        assert row.concept == concepts[code]
    assert sorted(held) == sorted(concepts)
    assert len(held) == len(set(held)) == len(OCCUPATION_CODES)
    assert all(LOW[i].code < LOW[i + 1].code for i in range(len(LOW) - 1))


def test_four_disability_items_map_yes_to_their_hpo_terms_and_two_map_none() -> None:
    mapped = {
        (row.when[0].field, row.concept, row.label)
        for row in DISABILITY_ROWS
        if isinstance(row.when[0], Codes) and row.when[0].codes == frozenset({1})
    }
    assert mapped == {
        ("DEAR", "HP:0000365", "Hearing impairment"),
        ("DEYE", "HP:0000505", "Visual impairment"),
        ("DREM", "HP:0100543", "Cognitive impairment"),
        ("DPHY", "HP:0001288", "Gait disturbance"),
    }
    fields = {clause.field for row in ACS_FIELD_ROWS for clause in row.when}
    assert "DDRS" not in fields
    assert "DOUT" not in fields
