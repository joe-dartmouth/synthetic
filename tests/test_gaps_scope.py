"""`bko.gaps.scope.load_scope` over releases served in-process and written to tmp_path."""

from __future__ import annotations

import logging

import pytest

from bko.gaps.errors import GapError
from bko.gaps.scope import load_scope
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import LADDER_HPO, LADDER_MONDO
from tests._ontology_fixtures import RoutingOpener


BODIES = {"HP": LADDER_HPO.encode(), "MONDO": LADDER_MONDO.encode()}
PINS = {"HP": pin(BODIES["HP"], name="hpo"), "MONDO": pin(BODIES["MONDO"], name="mondo")}
SERVED = {PINS[prefix].url: body for prefix, body in BODIES.items()}


def test_load_scope_refuses_an_unpinned_prefix(tmp_path):
    opener = RoutingOpener(SERVED)
    with pytest.raises(GapError, match="no pinned release for the prefix DOID"):
        load_scope({"DOID:1", "HP:9000003"}, tmp_path, releases=PINS, open_url=opener)
    assert opener.calls == []


def test_load_scope_logs_each_fragments_scope_under_bko_gaps(tmp_path, caplog):
    opener = RoutingOpener(SERVED)
    caplog.set_level(logging.INFO, logger="bko.gaps")
    ontologies, fragments = load_scope(
        {"HP:9000003", "MONDO:9000004"}, tmp_path, releases=PINS, open_url=opener
    )
    assert sorted(ontologies) == ["HP", "MONDO"]
    assert [fragment.name for fragment in fragments] == ["hpo", "mondo"]
    scopes = [
        (record.name, record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("scope ")
    ]
    assert scopes == [
        ("bko.gaps", "scope release=hpo classes=3 statements=2"),
        ("bko.gaps", "scope release=mondo classes=3 statements=2"),
    ]
