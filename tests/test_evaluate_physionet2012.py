"""The `physionet2012` dataset through `bko.evaluate.data` from a catalog entry with fixture pins,
and PhysioNet 2012's field rows.

No test reaches the API or reads `data/`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

import pytest

from bko.datasets.states import Cutoffs
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, dataset_releases, digest, field_rows, frame, load
from bko.ontology import physionet2012_rows
from bko.ontology.catalog import CATALOG
from bko.ontology.mapping import CONCEPT_ROWS
from bko.ontology.physionet2012_rows import PHYSIONET2012_FIELD_ROWS, state_row
from bko.ontology.schema import Codes
from tests._ontology_fixtures import RoutingOpener
from tests._physionet2012_fixtures import HR_CUTOFF, built, mixed


FILES = built()
SOURCES = tuple(source for source, _ in FILES)
BODIES = {source.url: data for source, data in FILES}
RECORDS = sum(len(held) for held in mixed().values())
"""The fixture's stays, every one a record: no age or sex is unknown."""


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context(opener=None) -> Context:
    catalogued = dataclasses.replace(CATALOG["physionet2012"], sources=SOURCES)
    return Context(
        open_url=RoutingOpener(BODIES) if opener is None else opener,
        catalog={"physionet2012": catalogued},
    )


def test_the_physionet2012_dataset_loads_and_logs_its_data_line(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.PHYSIONET2012, tmp_path, _context())
    assert Dataset.PHYSIONET2012.value == "physionet2012"
    assert data.frame.name == "physionet2012"
    assert tuple(data.records.columns) == data.frame.columns
    assert len(data.records) == RECORDS
    assert (data.field_rows, data.name, data.agreement, data.held) == (
        PHYSIONET2012_FIELD_ROWS,
        "physionet2012",
        None,
        None,
    )
    line = (
        f"data dataset=physionet2012 frame=physionet2012 records={RECORDS} "
        f"digest={digest(data.records)} agreement=none"
    )
    assert line in caplog.messages


def test_a_catalog_with_fixture_pins_fetches_nothing_else(tmp_path: Path) -> None:
    context = _context()
    load(Dataset.PHYSIONET2012, tmp_path, context)
    opener = context.open_url
    assert isinstance(opener, RoutingOpener)
    assert [url for url, _ in opener.calls] == [source.url for source in SOURCES]
    load(Dataset.PHYSIONET2012, tmp_path, _context(_refusing_opener))


def test_field_rows_and_releases_of_physionet2012_need_no_data() -> None:
    context = _context(_refusing_opener)
    assert field_rows(Dataset.PHYSIONET2012, context) == PHYSIONET2012_FIELD_ROWS
    assert dataset_releases(Dataset.PHYSIONET2012, context) is context.releases
    entry = CATALOG["physionet2012"]
    assert [source.filename for source in entry.sources] == [
        "set-a.zip",
        "set-b.zip",
        "set-c.tar.gz",
        "Outcomes-a.txt",
        "Outcomes-b.txt",
        "Outcomes-c.txt",
    ]
    assert entry.agreement is None
    assert entry.releases is None


def test_the_frame_of_physionet2012_needs_its_records() -> None:
    with pytest.raises(ValueError, match="physionet2012"):
        frame(Dataset.PHYSIONET2012)


def test_the_rows_map_these_states_to_these_concepts() -> None:
    mapped = []
    for row in PHYSIONET2012_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        mapped.append((clause.field, tuple(sorted(clause.codes)), row.concept, row.label))
    assert mapped == [
        ("hr_max", (2,), "HP:0001649", "Tachycardia"),
        ("resprate_max", (2,), "HP:0002789", "Tachypnea"),
        ("temp_max", (2,), "HP:0001945", "Fever"),
        ("temp_min", (1,), "HP:0002045", "Hypothermia"),
        ("wbc_max", (2,), "HP:0001974", "Increased total leukocyte count"),
        ("wbc_min", (1,), "HP:0001882", "Decreased total leukocyte count"),
        ("map_min", (1,), "HP:0002615", "Hypotension"),
        ("creatinine_max", (2,), "HP:0003259", "Elevated circulating creatinine concentration"),
        ("bilirubin_max", (2,), "HP:0002904", "Hyperbilirubinemia"),
        ("platelets_min", (1,), "HP:0001873", "Thrombocytopenia"),
        ("gcs_min", (1,), "HP:0004372", "Reduced consciousness"),
    ]


def test_every_cutoff_row_cites_singer_and_names_the_original() -> None:
    for row in PHYSIONET2012_FIELD_ROWS:
        assert row.source == "singer-2016", row
        assert "(read in singer-2016's reproduction)" in row.limits, row
        assert ("bone-1992" in row.limits) != ("vincent-1996" in row.limits), row
    assert PHYSIONET2012_FIELD_ROWS[0].limits == (
        f"hr_max above {HR_CUTOFF}, the stay's highest HR over its first 48 hours, a SIRS "
        "criterion of bone-1992 (read in singer-2016's reproduction); one reading, not a "
        "diagnosis; a stay that never measured it is state 0 and shows nothing"
    )
    assert "the stay's lowest MAP over" in PHYSIONET2012_FIELD_ROWS[6].limits


def test_the_rows_are_in_the_checked_table() -> None:
    assert set(PHYSIONET2012_FIELD_ROWS) <= set(CONCEPT_ROWS)


def test_a_row_on_cutoffs_that_cite_nothing_is_refused(monkeypatch) -> None:
    uncited = MappingProxyType({"hr_max": ("HR", "max", Cutoffs(above=(float(HR_CUTOFF),)))})
    monkeypatch.setattr(physionet2012_rows, "REDUCED", uncited)
    with pytest.raises(ValueError, match="hr_max's cutoffs cite no source"):
        state_row("hr_max", f"above {HR_CUTOFF}", "HP:0001649", "Tachycardia", "t")


def test_a_row_on_a_band_the_cutoffs_lack_is_refused() -> None:
    with pytest.raises(ValueError, match=f"no band 'above {HR_CUTOFF + 10}'"):
        state_row("hr_max", f"above {HR_CUTOFF + 10}", "HP:0001649", "Tachycardia", "t")
