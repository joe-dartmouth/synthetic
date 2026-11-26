"""`bko.ontology.occupations`: each `SOCP` code to its OccO class by SOC level, and the modules.

The release is the small OccO text of `tests._occupation_fixtures`; no test reads `data/`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
from collections import Counter
from pathlib import Path

import pytest

from bko.ontology.__main__ import Context, main
from bko.ontology.obo import parse_obo
from bko.ontology.occupations import REASONS, generate, part, render, soc_classes, soc_ids
from bko.ontology.schema import OccupationCode
from bko.ontology.terms import SOC_ID, Ontology
from tests._acs_fixtures import acs_archive
from tests._datasets_fixtures import pin
from tests._occupation_fixtures import CHIEF_EXECUTIVE, OCCO_TEXT
from tests._ontology_fixtures import FakeRobot, RoutingOpener, write_obo


def _occo(tmp_path, text: str = OCCO_TEXT) -> Ontology:
    return parse_obo(write_obo(tmp_path, text), name="occo")


def _rows(tmp_path, pairs, text: str = OCCO_TEXT) -> tuple[list[tuple[str, ...]], Counter[str]]:
    rows, unmapped = generate(pairs, _occo(tmp_path, text))
    return [(r.code, r.soc, r.concept, r.label) for r in rows], unmapped


def test_a_detailed_soc_code_maps_to_its_onet_00_class(tmp_path):
    rows, unmapped = _rows(tmp_path, {"0010": "111011", "0120": "113031"})
    assert rows == [
        ("0010", "11-1011.00", "OCCO:11101100", "chief executive"),
        ("0120", "11-3031", "OCCO:11303100", "financial manager"),
    ]
    assert unmapped == Counter({"no class": 0, "several classes": 0, "several SOC ids": 0})
    assert soc_ids("113031") == ("11-3031.00", "11-3031")


def test_a_five_digit_aggregate_maps_to_its_broad_class(tmp_path):
    rows, _ = _rows(tmp_path, {"1005": "15124X", "6240": "472040", "0011": "111010"})
    assert rows == [
        ("0011", "11-1010", "OCCO:11101000", "chief executive (broad)"),
        ("1005", "15-1240", "OCCO:15124000", "database or network administrator or architect"),
        ("6240", "47-2040", "OCCO:47204000", "carpet, floor, or tile installer or finisher"),
    ]


def test_a_shorter_aggregate_maps_to_its_minor_class(tmp_path):
    pairs = {"0010": "1110XX", "1240": "1520XX", "6260": "472XXX", "6270": "4720YY"}
    rows, _ = _rows(tmp_path, {**pairs, "0030": "113000"})
    assert rows == [
        ("0010", "11-1000", "OCCO:11100000", "top executive"),
        ("0030", "11-3000", "OCCO:11300000", "operations specialties manager"),
        ("1240", "15-2000", "OCCO:15200000", "mathematical science occupation"),
        ("6260", "47-2000", "OCCO:47200000", "construction trades worker"),
        ("6270", "47-2000", "OCCO:47200000", "construction trades worker"),
    ]


def test_a_code_with_no_class_is_counted_by_reason(tmp_path):
    twice = OCCO_TEXT + (
        "\n[Term]\nid: OCCO:47204001\nname: tile setter\nis_a: OCCO:47204000\n"
        'property_value: OCCO:00000008 "47-2040" xsd:string\n'
    )
    obsolete = CHIEF_EXECUTIVE + "is_obsolete: true\n"
    rows, unmapped = _rows(
        tmp_path,
        {"9830": "559830", "6240": "472040", "0010": "111011", "0020": "111021"},
        twice.replace(CHIEF_EXECUTIVE, obsolete),
    )
    assert rows == []
    assert REASONS == ("no class", "several classes", "several SOC ids")
    assert unmapped == Counter({"no class": 3, "several classes": 1, "several SOC ids": 0})


def test_a_class_carrying_two_soc_ids_is_refused_and_counted(tmp_path):
    stanza = CHIEF_EXECUTIVE + 'property_value: OCCO:00000008 "11-1011.03" xsd:string\n'
    rows, unmapped = _rows(
        tmp_path, {"0010": "111011", "0120": "113031"}, OCCO_TEXT.replace(CHIEF_EXECUTIVE, stanza)
    )
    assert rows == [("0120", "11-3031", "OCCO:11303100", "financial manager")]
    assert unmapped == Counter({"no class": 0, "several classes": 0, "several SOC ids": 1})


def test_a_class_carrying_one_soc_id_twice_counts_once(tmp_path):
    # pronto collapses a repeated literal, so the doubled term is built directly.
    occo = _occo(tmp_path)
    chief = occo.terms["OCCO:11101100"]
    doubled = dataclasses.replace(chief, annotations=chief.annotations * 2)
    assert doubled.annotation(SOC_ID) == ("11-1011.00", "11-1011.00")
    terms = {**occo.terms, chief.curie: doubled}
    rows, unmapped = generate({"0010": "111011"}, dataclasses.replace(occo, terms=terms))
    assert rows == [OccupationCode("0010", "11-1011.00", "OCCO:11101100", "chief executive")]
    assert unmapped == Counter({"no class": 0, "several classes": 0, "several SOC ids": 0})
    assert soc_classes(dataclasses.replace(occo, terms=terms))[chief.curie] == {"11-1011.00"}


@pytest.mark.parametrize("socp", ["11101", "1110111", "11XXXX", "1X1011", "11-101", ""])
def test_a_code_the_rule_cannot_read_is_refused(socp):
    with pytest.raises(ValueError, match="not a SOC code or aggregate the rule reads"):
        soc_ids(socp)


def test_a_chosen_class_with_no_name_is_refused(tmp_path):
    unnamed = OCCO_TEXT.replace("name: chief executive\n", "")
    with pytest.raises(ValueError, match="OCCO:11101100 has no name in occo"):
        generate({"0010": "111011"}, _occo(tmp_path, unnamed))


ROW = OccupationCode("0010", "11-1011.00", "OCCO:11101100", "chief executive")
HIGH = OccupationCode("9800", "55-1010", "OCCO:55101000", 'officer "special" leader')


def test_each_row_belongs_to_the_part_of_its_soc_major_group():
    assert part(ROW) == "low"
    assert part(dataclasses.replace(ROW, soc="29-2000")) == "low"
    assert part(dataclasses.replace(ROW, soc="31-1000")) == "high"


def test_a_rendered_module_holds_its_rows_one_per_line():
    text = render([HIGH], "high")
    assert text.startswith('"""ACS\'s occupation rows for SOC major groups from 31 up, generated')
    assert "`python -m bko.ontology occupations --part high` writes this module" in text
    assert '    ("9800", "55-1010", "OCCO:55101000", "officer \\"special\\" leader"),\n' in text
    namespace: dict[str, object] = {}
    exec(compile(text, "generated", "exec"), namespace)  # ruff: ignore[exec-builtin]
    assert namespace["OCCUPATION_CODES"] == (HIGH,)
    assert "below 31" in render([ROW], "low")


def test_a_row_of_the_other_part_or_with_a_control_character_is_refused():
    with pytest.raises(
        ValueError, match=r"OCCP '0010': SOC id 11-1011\.00 belongs to the low part"
    ):
        render([ROW], "high")
    for breaking in ("\n", "\x85", "\u2028", "\u2029"):
        with pytest.raises(ValueError, match="a line-breaking character"):
            render([dataclasses.replace(ROW, label=f"chief{breaking}executive")], "low")


def _command(tmp_path, part_name: str, opener=None):
    owl = dataclasses.replace(pin(b"<owl/>", name="occo"), filename="occo.owl")
    jar = pin(b"robot" * 50, name="robot-jar")
    archive = acs_archive()
    acs = pin(archive, name="acs-ma-2024")
    context = Context(
        open_url=opener
        or RoutingOpener({owl.url: b"<owl/>", jar.url: b"robot" * 50, acs.url: archive}),
        robot=jar,
        resolve_java=lambda: (Path("/opt/jdk/bin/java"), "openjdk"),
        robot_runner=FakeRobot(text=OCCO_TEXT),
        occo=owl,
        acs=acs,
    )
    out = io.StringIO()
    code = main(
        ["occupations", "--part", part_name, "--data-dir", str(tmp_path)], context=context, out=out
    )
    return code, out.getvalue()


def test_the_occupations_command_writes_one_parts_module(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code, text = _command(tmp_path, "low")
    assert code == 0
    assert '    ("0010", "11-1000", "OCCO:11100000", "top executive"),\n' in text
    assert '    ("6240", "47-2040", "OCCO:47204000",' not in text
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert 'occupations unmapped parts=all reason="no class" codes=2' in caplog.messages
    assert 'occupations unmapped parts=all reason="several classes" codes=0' in caplog.messages
    assert 'occupations unmapped parts=all reason="several SOC ids" codes=0' in caplog.messages
    # The archive holds four OCCP codes; the fixture's chief executive and eleven of `CLASSES`
    # carry a SOC id.
    assert (
        f"occupations part=low rows=1 of=2 codes=4 holders=12 sha256={digest} "
        "release=releases/2025-10-06" in caplog.messages
    )
    code, high = _command(tmp_path, "high")
    assert code == 0
    assert '    ("6240", "47-2040", "OCCO:47204000",' in high


def test_the_occupations_command_needs_a_part(tmp_path):
    with pytest.raises(SystemExit):
        main(["occupations", "--data-dir", str(tmp_path)])
