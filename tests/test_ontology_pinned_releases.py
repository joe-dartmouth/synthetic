"""The mapping table and Santos et al.'s example against the pinned MONDO, HPO, DO and OccO files.

Every other ontology test reads OBO text built in `tmp_path`, so a change that alters how a real
release parses passes `make check`. These cases load the releases from the data root and are
skipped unless `--releases` is passed, since the gate reads no `data/`. A release absent from the
data root fails them rather than skipping them, as does an OccO release not yet converted beside its
OWL file: they never fetch or run Java. `docs/DEVELOPMENT.md` (Ontologies) says when to run them and
how to fetch the releases first.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import BinaryIO

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import Source
from bko.ontology.__main__ import Context, check_releases
from bko.ontology.check import check
from bko.ontology.convert import converted
from bko.ontology.mapping import CHECKED_ROWS
from bko.ontology.obo import load_ontology
from bko.ontology.releases import load_releases
from bko.ontology.santos import PAPER_RELEASES, reproduce
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _parsed(source: Source) -> Ontology:
    return load_ontology(source, default_data_dir())


def _load(sources: Mapping[str, Source]) -> dict[str, Ontology]:
    return {prefix: _parsed(source) for prefix, source in sources.items()}


def _no_fetch(url: str, timeout: float) -> BinaryIO:
    msg = f"a release test fetched {url} ({timeout} s)"
    raise AssertionError(msg)


def _no_java() -> tuple[Path, str]:
    msg = "a release test ran Java: the converted OccO is absent from the data root"
    raise AssertionError(msg)


def test_the_table_checks_clean_against_the_pinned_releases() -> None:
    releases = check_releases(Context())
    assert sorted(releases) == ["HP", "MONDO", "OCCO"]
    convert = partial(converted, resolve_java=_no_java)
    loaded = load_releases(releases, default_data_dir(), open_url=_no_fetch, convert=convert)
    verdicts = check(CHECKED_ROWS, loaded)
    assert len(verdicts) == len(CHECKED_ROWS)
    assert [
        f"{v.table} {v.concept or v.subject}: {v.problem}"
        for v in verdicts
        if v.problem is not None
    ] == []


def test_the_reproduction_differs_nowhere_against_the_pinned_releases() -> None:
    releases = _load(PAPER_RELEASES)
    findings = reproduce(releases["MONDO"], releases["DOID"])
    assert any(f.status == "match" for f in findings)
    assert [
        f"{f.kind} {f.item}: paper {f.paper!r}, measured {f.measured!r}"
        for f in findings
        if f.status == "differs"
    ] == []
