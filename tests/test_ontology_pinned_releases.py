"""The mapping table and Santos et al.'s example against the pinned MONDO, HPO and DO releases.

Every other ontology test reads OBO text built in `tmp_path`, so a change that alters how a real
release parses passes `make check`. These cases load the releases from the data root and are
skipped unless `--releases` is passed, since the gate reads no `data/`. A release absent from the
data root fails them rather than skipping them. `docs/DEVELOPMENT.md` (Ontologies) says when to
run them and how to fetch the releases first.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import Source
from bko.ontology.check import check
from bko.ontology.mapping import CHECKED_ROWS, RELEASES
from bko.ontology.obo import load_ontology
from bko.ontology.santos import PAPER_RELEASES, reproduce
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _parsed(source: Source) -> Ontology:
    return load_ontology(source, default_data_dir())


def _load(sources: Mapping[str, Source]) -> dict[str, Ontology]:
    return {prefix: _parsed(source) for prefix, source in sources.items()}


def test_the_table_checks_clean_against_the_pinned_releases() -> None:
    verdicts = check(CHECKED_ROWS, _load(RELEASES))
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
