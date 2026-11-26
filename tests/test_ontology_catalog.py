"""The dataset catalog: each entry's pins, loader, frame and rows, read with fixture pins."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.datasets.frames import digest
from bko.datasets.sources import SUPPORT2, Agreement
from bko.datasets.support2 import load_support2
from bko.datasets.support2_frame import support2_frame
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


def test_the_catalog_holds_support2() -> None:
    assert list(CATALOG) == ["support2"]
    entry = CATALOG["support2"]
    assert entry.name == "support2"
    assert entry.sources == (SUPPORT2,)
    assert entry.load is load_support2
    assert entry.frame is support2_frame
    assert entry.rows == SUPPORT2_FIELD_ROWS
    assert entry.releases is None
    assert entry.agreement is None


def test_read_fetches_each_source_then_loads_and_checks(tmp_path: Path) -> None:
    entry = dataclasses.replace(CATALOG["support2"], sources=(PIN,))
    opener = RoutingOpener({PIN.url: ARCHIVE})
    records, frame = entry.read(tmp_path, open_url=opener)
    assert [url for url, _ in opener.calls] == [PIN.url]
    assert frame.name == "support2"
    assert tuple(records.columns) == frame.columns
    assert len(records) == 12
    assert digest(records) == "b82d2872761d699238d7045ebb7485b05449640f33b5755a8c335c3fcf2177a9"
    again, _ = entry.read(tmp_path, open_url=_refusing_opener)
    assert again.equals(records)


def test_read_refuses_records_the_frame_does_not_hold(tmp_path: Path) -> None:
    def stray(data_dir: Path, *, sources) -> pd.DataFrame:
        return load_support2(data_dir, sources=sources).assign(race=9)

    entry = dataclasses.replace(CATALOG["support2"], sources=(PIN,), load=stray)
    with pytest.raises(ContractError, match=r"race holds codes \[9\]"):
        entry.read(tmp_path, open_url=RoutingOpener({PIN.url: ARCHIVE}))


def test_an_entry_with_no_source_or_mixed_agreements_is_refused() -> None:
    entry = CATALOG["support2"]
    with pytest.raises(ValueError, match="names no source"):
        dataclasses.replace(entry, sources=())
    held = dataclasses.replace(PIN, name="held", agreement=Agreement("dua"))
    with pytest.raises(ValueError, match="different agreements"):
        dataclasses.replace(entry, sources=(PIN, held))
    assert dataclasses.replace(entry, sources=(held,)).agreement == Agreement("dua")


def test_resolved_releases_are_the_entrys_own_else_the_default() -> None:
    default = {"HP": dataclasses.replace(PIN, name="hpo")}
    own = {"MONDO": dataclasses.replace(PIN, name="mondo")}
    entry = CATALOG["support2"]
    assert entry.resolved_releases(default) is default
    assert dataclasses.replace(entry, releases=own).resolved_releases(default) is own
