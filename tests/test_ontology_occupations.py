"""ACS occupation rows against an OccO release: their SOC id check, and the check command's
conversion of an OWL-only release pinned by a catalog entry.

Java never runs: `FakeRobot` writes the converted OBO.
"""

from __future__ import annotations

import dataclasses
import io
import logging
from pathlib import Path
from typing import Final

import pytest

from bko.datasets.fetch import snapshot_dir
from bko.ontology.__main__ import Context, check_releases, main
from bko.ontology.catalog import CATALOG
from bko.ontology.check import check, prefixes
from bko.ontology.obo import parse_obo
from bko.ontology.schema import OccupationCode, UnmappedCode
from bko.ontology.terms import Ontology
from tests._datasets_fixtures import pin
from tests._occupation_fixtures import CHIEF_EXECUTIVE, OCCO_TEXT
from tests._ontology_fixtures import FakeRobot, RoutingOpener, write_obo


JAVA: Final = Path("/opt/jdk/bin/java")
OWL: Final = b'<?xml version="1.0"?>\n<rdf:RDF><owl:Ontology rdf:about="occo"/></rdf:RDF>\n'
JAR: Final = b"robot" * 50
CHIEF: Final = OccupationCode("0010", "11-1011.00", "OCCO:11101100", "chief executive")


def _java() -> tuple[Path, str]:
    return JAVA, 'openjdk version "21.0.4"'


def _occo(tmp_path: Path) -> Ontology:
    return parse_obo(write_obo(tmp_path, OCCO_TEXT), name="occo")


def test_an_occupation_row_whose_soc_id_differs_fails_the_check(tmp_path):
    other = dataclasses.replace(CHIEF, code="0011", soc="11-1010")
    unannotated = OccupationCode("9999", "11-0000", "OCCO:00000001", "occupation holder")
    verdicts = check((CHIEF, other, unannotated), {"OCCO": _occo(tmp_path)})
    assert [(v.table, v.subject, v.problem) for v in verdicts] == [
        ("occupation-code", "OCCP:0010", None),
        (
            "occupation-code",
            "OCCP:0011",
            (
                "occo releases/2025-10-06 gives OCCO:11101100 OCCO:00000008 11-1011.00; "
                "the row says 11-1010"
            ),
        ),
        (
            "occupation-code",
            "OCCP:9999",
            "occo releases/2025-10-06 gives OCCO:00000001 OCCO:00000008 none; the row says 11-0000",
        ),
    ]


def test_an_occupation_row_whose_class_carries_two_soc_ids_fails_the_check(tmp_path):
    stanza = CHIEF_EXECUTIVE + 'property_value: OCCO:00000008 "11-1011.03" xsd:string\n'
    occo = parse_obo(write_obo(tmp_path, OCCO_TEXT.replace(CHIEF_EXECUTIVE, stanza)), name="occo")
    [verdict] = check((CHIEF,), {"OCCO": occo})
    assert verdict.problem == (
        "occo releases/2025-10-06 gives OCCO:11101100 OCCO:00000008 11-1011.00, 11-1011.03; "
        "the row says 11-1011.00"
    )


def test_an_occupation_row_whose_class_repeats_its_soc_id_passes_the_check(tmp_path):
    occo = _occo(tmp_path)
    chief = occo.terms[CHIEF.concept]
    doubled = dataclasses.replace(chief, annotations=chief.annotations * 2)
    terms = {**occo.terms, CHIEF.concept: doubled}
    [verdict] = check((CHIEF,), {"OCCO": dataclasses.replace(occo, terms=terms)})
    assert verdict.problem is None


def test_an_occupation_row_is_checked_for_its_name_first(tmp_path):
    renamed = dataclasses.replace(CHIEF, label="chief executives")
    [verdict] = check((renamed,), {"OCCO": _occo(tmp_path)})
    assert verdict.problem == "named 'chief executive' in occo releases/2025-10-06"


def test_the_rows_name_the_prefixes_the_check_loads():
    excluded = UnmappedCode("401314000", "x", "test", "mondo")
    assert prefixes((CHIEF, excluded)) == frozenset({"OCCO", "MONDO"})


def _context(owl_source, jar, robot, catalog=None) -> Context:
    entry = dataclasses.replace(CATALOG["support2"], releases={"OCCO": owl_source})
    return Context(
        open_url=RoutingOpener({owl_source.url: OWL, jar.url: JAR}),
        releases={},
        rows=(CHIEF,),
        robot=jar,
        resolve_java=_java,
        robot_runner=robot,
        catalog={"toy": entry} if catalog is None else catalog,
    )


def test_the_check_converts_an_owl_only_release_first(tmp_path, caplog):
    owl_source = dataclasses.replace(pin(OWL, name="occo"), filename="occo.owl")
    jar = pin(JAR, name="robot-jar")
    robot = FakeRobot(text=OCCO_TEXT)
    out = io.StringIO()
    context = _context(owl_source, jar, robot)
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code = main(["check", "--data-dir", str(tmp_path)], context=context, out=out)
    assert code == 0
    assert out.getvalue() == (
        "table,subject,concept,label,status\n"
        "occupation-code,OCCP:0010,OCCO:11101100,chief executive,ok\n"
    )
    assert len(robot.calls) == 1
    assert robot.calls[0][robot.calls[0].index("--input") + 1].endswith("occo.owl")
    converted = snapshot_dir(owl_source, tmp_path) / f"occo.robot-{jar.sha256[:16]}.obo"
    assert converted.read_text(encoding="utf-8") == OCCO_TEXT
    assert "release occo data_version=releases/2025-10-06" in caplog.text
    assert "checked 1 rows: 0 disagree" in caplog.text


def test_the_commands_own_release_wins_over_a_catalog_entrys():
    own = pin(b"own", name="own")
    entry = dataclasses.replace(CATALOG["support2"], releases={"OCCO": pin(b"x", name="x")})
    context = Context(releases={"OCCO": own}, rows=(CHIEF,), catalog={"toy": entry})
    assert check_releases(context) == {"OCCO": own}


def test_a_prefix_no_release_pins_is_left_for_its_rows_to_report():
    context = Context(releases={}, rows=(CHIEF,), catalog={})
    assert check_releases(context) == {}


def test_two_catalog_entries_pinning_a_prefix_differently_are_refused():
    entries = {
        name: dataclasses.replace(CATALOG["support2"], releases={"OCCO": pin(body, name=name)})
        for name, body in (("first", b"1"), ("second", b"2"))
    }
    context = Context(releases={}, rows=(CHIEF,), catalog=entries)
    with pytest.raises(ValueError, match="prefix OCCO to different releases: first, second"):
        check_releases(context)
