"""`bko.gaps.scope.load_scope` over releases served in-process and written to tmp_path."""

from __future__ import annotations

import logging
from typing import Final

import pytest

from bko.datasets.split import SPLIT_SEED
from bko.gaps.errors import GapError
from bko.gaps.scope import load_scope, load_uci_scope
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ANGINA, LADDER_HPO, LADDER_MONDO, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, uci
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


UCI_E1: Final = uci({(40, 0): "4.0"}, "1.0", {**SIZES, (40, 0, 1): 5})
PIN: Final = pin(UCI_E1)


def test_load_uci_scope_splits_loads_and_observes(tmp_path, caplog):
    pins, served = ladder_releases()
    opener = RoutingOpener({**served, PIN.url: UCI_E1})
    caplog.set_level(logging.INFO)
    scope = load_uci_scope(
        tmp_path, uci=PIN, releases=pins, field_rows=(ANGINA, STENOSIS), open_url=opener
    )
    assert scope.parts.seed == SPLIT_SEED
    assert set(scope.ontologies) == {"HP", "MONDO"}
    assert scope.cells.whole.name == PIN.name
    assert f"fingerprint={scope.parts.fingerprint}" in caplog.text
