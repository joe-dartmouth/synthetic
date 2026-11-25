"""The `support2` dataset through `bko.evaluate.data`, read from a catalog entry with fixture pins.

No test reaches the API or reads `data/`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import BinaryIO

import pytest

from bko.datasets.sources import Source
from bko.datasets.support2 import load_support2
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, dataset_releases, digest, field_rows, frame, load
from bko.ontology.catalog import CATALOG
from bko.ontology.support2_rows import SUPPORT2_FIELD_ROWS
from tests._datasets_fixtures import pin
from tests._ontology_fixtures import RoutingOpener
from tests._support2_fixtures import support2_archive


ARCHIVE = support2_archive()
PIN = pin(ARCHIVE, "support2")


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _context(opener=None, **entry) -> Context:
    catalogued = dataclasses.replace(CATALOG["support2"], sources=(PIN,), **entry)
    return Context(
        open_url=RoutingOpener({PIN.url: ARCHIVE}) if opener is None else opener,
        catalog={"support2": catalogued},
    )


def test_the_support2_dataset_loads_and_logs_its_data_line(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.SUPPORT2, tmp_path, _context())
    assert Dataset.SUPPORT2.value == "support2"
    assert data.frame.name == "support2"
    assert tuple(data.records.columns) == data.frame.columns
    assert data.records.equals(load_support2(tmp_path, sources=(PIN,)))
    assert (data.field_rows, data.name, data.agreement, data.held) == (
        SUPPORT2_FIELD_ROWS,
        "support2",
        None,
        None,
    )
    line = (
        f"data dataset=support2 frame=support2 records={len(data.records)} "
        f"digest={digest(data.records)} agreement=none"
    )
    assert line in caplog.messages


def test_field_rows_of_support2_need_no_data() -> None:
    assert field_rows(Dataset.SUPPORT2, _context(_refusing_opener)) == SUPPORT2_FIELD_ROWS


def test_field_rows_and_releases_follow_the_contexts_catalog() -> None:
    own = {"HP": PIN}
    context = _context(_refusing_opener, rows=SUPPORT2_FIELD_ROWS[:1], releases=own)
    assert field_rows(Dataset.SUPPORT2, context) == SUPPORT2_FIELD_ROWS[:1]
    assert dataset_releases(Dataset.SUPPORT2, context) == own
    plain = _context(_refusing_opener)
    assert dataset_releases(Dataset.SUPPORT2, plain) == plain.releases


def test_a_catalog_with_a_fixture_pin_fetches_nothing_else(tmp_path: Path) -> None:
    context = _context()
    load(Dataset.SUPPORT2, tmp_path, context)
    opener = context.open_url
    assert isinstance(opener, RoutingOpener)
    assert [url for url, _ in opener.calls] == [PIN.url]
    load(Dataset.SUPPORT2, tmp_path, _context(_refusing_opener))


def test_the_frame_of_support2_needs_its_records() -> None:
    with pytest.raises(ValueError, match="support2"):
        frame(Dataset.SUPPORT2)


def test_the_default_context_reads_the_real_catalog() -> None:
    assert Context().catalog is CATALOG
    source: Source = CATALOG["support2"].sources[0]
    assert source.name == "support2"
