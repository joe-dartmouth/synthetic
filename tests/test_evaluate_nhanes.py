"""The `nhanes` dataset through `bko.evaluate.data` from a catalog entry with fixture pins, and
NHANES's field rows.

No test reaches the API or reads `data/`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import BinaryIO, Final

import pytest

from bko.datasets.nhanes import CITED, COMPONENTS, TERTILES
from bko.datasets.states import band_meanings
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, dataset_releases, digest, field_rows, frame, load
from bko.ontology.catalog import CATALOG
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from bko.ontology.nhanes_rows import NHANES_FIELD_ROWS, state_row
from bko.ontology.schema import Codes
from tests._datasets_fixtures import pin
from tests._nhanes_fixtures import MIXED_DIGEST, files, mixed
from tests._ontology_fixtures import RoutingOpener
from tests._registry import REGISTRY, read, registry_entries


FILES = files(mixed())
PINS = tuple(pin(FILES[name], f"nhanes-{name.lower()}") for name in COMPONENTS)


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context(opener=None, **entry) -> Context:
    catalogued = dataclasses.replace(CATALOG["nhanes"], sources=PINS, **entry)
    served = {source.url: FILES[name] for name, source in zip(COMPONENTS, PINS, strict=True)}
    return Context(
        open_url=RoutingOpener(served) if opener is None else opener,
        catalog={"nhanes": catalogued},
    )


def test_the_nhanes_dataset_loads_and_logs_its_data_line(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.NHANES, tmp_path, _context())
    assert data.frame.name == "nhanes"
    assert tuple(data.records.columns) == data.frame.columns
    assert len(data.records) == 9
    assert digest(data.records) == MIXED_DIGEST
    assert (data.field_rows, data.name, data.agreement, data.held) == (
        NHANES_FIELD_ROWS,
        "nhanes",
        None,
        None,
    )
    line = f"data dataset=nhanes frame=nhanes records=9 digest={MIXED_DIGEST} agreement=none"
    assert line in caplog.messages


def test_a_catalog_with_fixture_pins_fetches_each_component_once(tmp_path: Path) -> None:
    context = _context()
    load(Dataset.NHANES, tmp_path, context)
    opener = context.open_url
    assert isinstance(opener, RoutingOpener)
    assert [url for url, _ in opener.calls] == [source.url for source in PINS]
    load(Dataset.NHANES, tmp_path, _context(_refusing_opener))


def test_field_rows_of_nhanes_need_no_data() -> None:
    assert field_rows(Dataset.NHANES, _context(_refusing_opener)) == NHANES_FIELD_ROWS


def test_nhanes_reads_the_contexts_releases_and_its_seven_pins() -> None:
    context = _context(_refusing_opener)
    assert dataset_releases(Dataset.NHANES, context) is context.releases
    entry = CATALOG["nhanes"]
    assert [source.filename for source in entry.sources] == [f"{n}.xpt" for n in COMPONENTS]
    assert {source.agreement for source in entry.sources} == {None}
    assert entry.releases is None


def test_the_frame_of_nhanes_needs_its_records() -> None:
    with pytest.raises(ValueError, match="nhanes"):
        frame(Dataset.NHANES)


def test_the_rows_map_these_bands_to_these_concepts() -> None:
    found = []
    for row in NHANES_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        bands = [band_meanings(CITED[clause.field])[code] for code in sorted(clause.codes)]
        found.append((clause.field, bands, row.concept, row.label, row.source))
    assert found == [
        (
            "BPXOSY",
            ["at least 130"],
            "HP:0004421",
            "Elevated systolic blood pressure",
            "muntner-2018",
        ),
        (
            "BPXODI",
            ["at least 80"],
            "HP:0005117",
            "Elevated diastolic blood pressure",
            "muntner-2018",
        ),
        (
            "LBXGH",
            ["at least 5.7 and below 6.5", "at least 6.5"],
            "HP:0040217",
            "Elevated hemoglobin A1c",
            "ada-2026",
        ),
        ("LBXTC", ["at least 240"], "HP:0003124", "Hypercholesterolemia", "nhlbi-2001"),
        (
            "LBDHDD",
            ["below 40"],
            "HP:0003233",
            "Decreased circulating HDL-C concentration",
            "nhlbi-2001",
        ),
    ]


READ_IN: Final = {
    "BPXOSY": ("muntner-2018", "whelton-2018"),
    "BPXODI": ("muntner-2018", "whelton-2018"),
    "LBXGH": ("ada-2026", None),
    "LBXTC": ("nhlbi-2001", "ncep-2001"),
    "LBDHDD": ("nhlbi-2001", "ncep-2001"),
}
"""Each row's field, the registry key of the record its cutoffs are read in, and the original
that record reproduces, None when the record is the original (ADR `research/nhanes-frame`)."""


def _names(text: str, key: str) -> bool:
    """Whether `text` holds `key` as a whole token, not inside a longer key."""
    return re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", text) is not None


def test_every_cutoff_row_cites_a_registered_source_and_names_its_original() -> None:
    keys = {key for _, key, _ in registry_entries(read(REGISTRY))}
    originals = {original for _, original in READ_IN.values() if original is not None}
    assert originals <= keys
    found = {}
    for row in NHANES_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        source, original = READ_IN[clause.field]
        assert row.source == source, row
        assert source in keys, row
        found[clause.field] = row.source
        for other in originals:
            assert _names(row.limits, other) == (other == original), (row, other)
    assert found == {field: source for field, (source, _) in READ_IN.items()}
    assert not _names("whelton-2018x and nhanes-2024", "whelton-2018")
    assert "nhanes-2024" in keys


def test_no_row_reads_bmi_or_state_0() -> None:
    for row in NHANES_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        assert clause.field not in TERTILES
        assert 0 not in clause.codes, row


@pytest.mark.parametrize(
    ("field", "band", "match"),
    [
        ("BMXBMI", "below 25.9", "^BMXBMI's cutoffs cite no source, so no row maps its states$"),
        ("LBXTC", "at least 200", "^no band 'at least 200' among "),
    ],
    ids=["tertile-field", "missing-band"],
)
def test_state_row_refuses_an_uncited_field_or_a_missing_band(
    field: str, band: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        state_row(field, (band,), "HP:0000001", "All", "a criterion")


def test_the_rows_join_the_checked_table() -> None:
    start = CONCEPT_ROWS.index(NHANES_FIELD_ROWS[0])
    assert CONCEPT_ROWS[start : start + len(NHANES_FIELD_ROWS)] == NHANES_FIELD_ROWS
    assert set(NHANES_FIELD_ROWS) <= set(CHECKED_ROWS)
