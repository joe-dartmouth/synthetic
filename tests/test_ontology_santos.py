"""`bko.ontology.santos`: Santos et al.'s example rebuilt from release text built in `tmp_path`."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pytest

from bko.bkb.fusion import fuse, is_source
from bko.ontology.obo import parse_obo
from bko.ontology.santos import WEIGHTS, Finding, paper_fragments, reproduce
from bko.ontology.terms import Ontology
from tests._ontology_fixtures import (
    DOID_9473_STANZA,
    SANTOS_DO_TEXT,
    SANTOS_MONDO_TEXT,
    SANTOS_REPORT,
    write_obo,
)


pytestmark = pytest.mark.filterwarnings("error")

EXTRA_EDGE_FROM = "is_a: MONDO:0001397 ! mononeuropathy\nproperty_value: skos:exactMatch DOID:11446"
EXTRA_EDGE_TO = (
    "is_a: MONDO:0001397 ! mononeuropathy\nis_a: MONDO:0021166 ! inflammatory disease\n"
    "property_value: skos:exactMatch DOID:11446"
)


def releases(
    directory: Path, mondo: str = SANTOS_MONDO_TEXT, do: str = SANTOS_DO_TEXT
) -> tuple[Ontology, Ontology]:
    return (
        parse_obo(write_obo(directory, mondo, "mondo.obo"), name="mondo"),
        parse_obo(write_obo(directory, do, "doid.obo"), name="do"),
    )


def differing(findings: tuple[Finding, ...]) -> set[tuple[str, str]]:
    return {(f.kind, f.item) for f in findings if f.status == "differs"}


def test_the_example_reproduces_with_no_differing_quantity(tmp_path):
    findings = reproduce(*releases(tmp_path))
    rows = [[f.kind, f.item, f.paper, f.measured, f.status] for f in findings]
    assert rows == list(csv.reader(io.StringIO(SANTOS_REPORT)))[1:]
    assert not differing(findings)


def test_the_fragments_hold_five_four_and_eight_snodes(tmp_path):
    fragments = paper_fragments(*releases(tmp_path))
    assert [(f.name, len(f.snodes)) for f in fragments] == [("mondo", 5), ("do", 4), ("bridge", 8)]


def test_only_every_fragment_normalization_gives_the_papers_third(tmp_path):
    measured = {(f.kind, f.item): f.measured for f in reproduce(*releases(tmp_path))}
    assert measured["inference", "a"] == repr(1 / 3)
    assert measured["inference", "a under pybkb normalization"] == "0.5"


def test_the_fused_example_is_not_mutex_where_mondo_gives_mononeuropathy_two_children(tmp_path):
    fused = fuse(paper_fragments(*releases(tmp_path)), WEIGHTS)
    pair = fused.non_mutex_pair()
    assert pair is not None
    first, second = (fused.snodes[index] for index in pair)
    assert first.head == second.head == ("MONDO:0001397", "a")
    antecedents = {c for snode in (first, second) for c, _ in snode.tail if not is_source(c)}
    assert antecedents == {"MONDO:0006960", "MONDO:0002121"}


def test_fusing_the_example_logs_its_size(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.bkb.fusion"):
        reproduce(*releases(tmp_path))
    assert "fused fragments=3 normalization=fragments inodes=27 snodes=33" in caplog.text


def test_a_term_table_1_printed_rightly_is_no_erratum(tmp_path):
    do = SANTOS_DO_TEXT + "\n[Term]\nid: DOID:114466\nname: sciatic neuropathy\n"
    findings = reproduce(*releases(tmp_path, do=do))
    assert differing(findings) == {("term", "DOID:114466")}
    [term] = [f for f in findings if f.item == "DOID:114466"]
    assert term.measured == "DOID:114466 present; DOID:11446 sciatic neuropathy"


def test_a_missing_class_differs_without_stopping_the_report(tmp_path):
    do = SANTOS_DO_TEXT.replace(DOID_9473_STANZA, "").replace(
        "is_a: DOID:9473 ! mononeuritis of lower limb\n", ""
    )
    assert differing(reproduce(*releases(tmp_path, do=do))) == {
        ("term", "DOID:9473"),
        ("pta", "TD2"),
        ("pta", "TD3"),
        ("instantiation", "pass 2"),
        ("instantiation", "after pass 2"),
        ("instantiation", "fixpoint"),
        ("chain", "fused"),
    }


def test_an_edge_beyond_the_paper_is_extra_and_breaks_the_emergent_claim(tmp_path):
    mondo = SANTOS_MONDO_TEXT.replace(EXTRA_EDGE_FROM, EXTRA_EDGE_TO)
    assert differing(reproduce(*releases(tmp_path, mondo=mondo))) == {
        ("pta", "extra"),
        ("instantiation", "pass 2"),
        ("instantiation", "after pass 2"),
        ("instantiation", "fixpoint"),
        ("chain", "mondo fragment"),
        ("chain", "mondo release"),
    }
