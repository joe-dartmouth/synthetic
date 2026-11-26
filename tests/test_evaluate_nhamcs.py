"""The `nhamcs` dataset through `bko.evaluate.data` from a catalog entry with a fixture pin, and
NHAMCS's field rows.

No test reaches the API or reads `data/`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import BinaryIO

import pytest

from bko.datasets.icd10_tokens import UNKNOWN, token_number
from bko.datasets.nhamcs import DIAGNOSES, FLAGS
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, dataset_releases, digest, field_rows, frame, load
from bko.ontology.catalog import CATALOG
from bko.ontology.icd10cm_codes import ICD10CM_CODES
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from bko.ontology.nhamcs_rows import DIAGNOSIS_ROWS, FLAG_ROWS, NHAMCS_FIELD_ROWS
from bko.ontology.schema import Codes
from tests._datasets_fixtures import pin
from tests._nhamcs_fixtures import nhamcs_archive
from tests._ontology_fixtures import RoutingOpener


ARCHIVE = nhamcs_archive()
PIN = pin(ARCHIVE, "nhamcs-ed-2022")
MIXED_DIGEST = "f878ed8d0354a333cbd904c7f6d5497a79745e65a4708fa5f19922ebd76014ce"
"""The sha256 of `mixed()`'s records as `bko.evaluate.data.digest` renders them."""


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context(opener=None, **entry) -> Context:
    catalogued = dataclasses.replace(CATALOG["nhamcs"], sources=(PIN,), **entry)
    return Context(
        open_url=RoutingOpener({PIN.url: ARCHIVE}) if opener is None else opener,
        catalog={"nhamcs": catalogued},
    )


def test_the_nhamcs_dataset_loads_and_logs_its_data_line(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.NHAMCS, tmp_path, _context())
    assert Dataset.NHAMCS.value == "nhamcs"
    assert data.frame.name == "nhamcs"
    assert tuple(data.records.columns) == data.frame.columns
    assert len(data.records) == 12
    assert digest(data.records) == MIXED_DIGEST
    assert (data.field_rows, data.name, data.agreement, data.held) == (
        NHAMCS_FIELD_ROWS,
        "nhamcs",
        None,
        None,
    )
    line = f"data dataset=nhamcs frame=nhamcs records=12 digest={MIXED_DIGEST} agreement=none"
    assert line in caplog.messages


def test_a_catalog_with_a_fixture_pin_fetches_nothing_else(tmp_path: Path) -> None:
    context = _context()
    load(Dataset.NHAMCS, tmp_path, context)
    opener = context.open_url
    assert isinstance(opener, RoutingOpener)
    assert [url for url, _ in opener.calls] == [PIN.url]
    load(Dataset.NHAMCS, tmp_path, _context(_refusing_opener))


def test_field_rows_and_releases_of_nhamcs_need_no_data() -> None:
    context = _context(_refusing_opener)
    assert field_rows(Dataset.NHAMCS, context) == NHAMCS_FIELD_ROWS
    assert dataset_releases(Dataset.NHAMCS, context) is context.releases
    entry = CATALOG["nhamcs"]
    assert [source.filename for source in entry.sources] == ["ed2022.zip"]
    assert entry.agreement is None
    assert entry.releases is None


def test_the_frame_of_nhamcs_needs_its_records() -> None:
    with pytest.raises(ValueError, match="nhamcs"):
        frame(Dataset.NHAMCS)


def test_each_diagnosis_columns_rows_partition_its_mapped_codes() -> None:
    mapped = {token_number(row.token): row.concept for row in ICD10CM_CODES}
    assert len(mapped) == len(ICD10CM_CODES)
    assert UNKNOWN not in mapped
    assert len(DIAGNOSIS_ROWS) == len(DIAGNOSES) * len(ICD10CM_CODES)
    for column in DIAGNOSES:
        held: list[int] = []
        for row in DIAGNOSIS_ROWS:
            [clause] = row.when
            assert isinstance(clause, Codes)
            if clause.field == column:
                [number] = clause.codes
                assert mapped[number] == row.concept, row
                held.append(number)
        assert len(held) == len(set(held)) == len(mapped), column
        assert set(held) == set(mapped), column
    assert mapped[token_number("A46-")] == "MONDO:0001266"


def test_the_flag_rows_map_these_items_on_code_1() -> None:
    found = []
    for row in FLAG_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        assert clause.codes == {1}
        assert row.source == "nhamcs-nd"
        found.append((clause.field, row.concept, row.label))
    assert found == [
        ("ALZHD", "MONDO:0001627", "dementia"),
        ("ASTHMA", "MONDO:0004979", "asthma"),
        ("CANCER", "MONDO:0004992", "cancer"),
        ("CEBVD", "MONDO:0011057", "cerebrovascular disorder"),
        ("CKD", "MONDO:0005300", "chronic kidney disease"),
        ("COPD", "MONDO:0005002", "chronic obstructive pulmonary disease"),
        ("CHF", "MONDO:0005009", "congestive heart failure"),
        ("DEPRN", "MONDO:0002050", "depressive disorder"),
        ("DIABTYP1", "MONDO:0005147", "type 1 diabetes mellitus"),
        ("DIABTYP2", "MONDO:0005148", "type 2 diabetes mellitus"),
        ("DIABTYP0", "MONDO:0005015", "diabetes mellitus"),
        ("ESRD", "MONDO:0004375", "end stage renal failure"),
        ("EDHIV", "MONDO:0005109", "HIV infectious disease"),
        ("HYPLIPID", "MONDO:0021187", "hyperlipidemia"),
        ("HTN", "HP:0000822", "Hypertension"),
        ("OBESITY", "MONDO:0011122", "obesity disorder"),
        ("OSA", "MONDO:0007147", "obstructive sleep apnea syndrome"),
        ("OSTPRSIS", "MONDO:0005298", "osteoporosis"),
    ]
    unmapped = set(FLAGS) - {field for field, _, _ in found}
    assert unmapped == {"ETOHAB", "CAD", "HPE", "SUBSTAB"}


def test_the_flag_rows_join_the_checked_table_and_the_diagnosis_rows_do_not() -> None:
    start = CONCEPT_ROWS.index(FLAG_ROWS[0])
    assert CONCEPT_ROWS[start : start + len(FLAG_ROWS)] == FLAG_ROWS
    assert set(FLAG_ROWS) <= set(CHECKED_ROWS)
    assert set(DIAGNOSIS_ROWS).isdisjoint(CHECKED_ROWS)
