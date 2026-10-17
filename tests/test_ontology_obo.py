"""`bko.ontology.obo` parses OBO text into standard-library values and refuses what it cannot."""

from __future__ import annotations

import logging
import pathlib
import urllib.request
import warnings
from typing import Any

import pytest

from bko.datasets.errors import ContractError, UnavailableError
from bko.ontology import obo
from bko.ontology.errors import ReleaseFormatError
from bko.ontology.obo import load_ontology, parse_obo
from bko.ontology.terms import Term
from tests._datasets_fixtures import pin, plant
from tests._ontology_fixtures import HPO_TEXT, MONDO_TEXT, write_obo


pytestmark = pytest.mark.filterwarnings("error")

HEADER = "format-version: 1.2\ndata-version: test/1\nontology: test\n"


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        msg = "the loader reached the network"
        raise AssertionError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _refused(tmp_path: pathlib.Path, text: str) -> str:
    path = write_obo(tmp_path, text)
    with pytest.raises(ReleaseFormatError) as caught:
        parse_obo(path, name="test")
    message = str(caught.value)
    assert str(path) in message
    return message


def test_real_stanza_parses_to_term(tmp_path):
    ontology = parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo")
    assert ontology.name == "mondo"
    assert ontology.terms["MONDO:0024644"] == Term(
        curie="MONDO:0024644",
        name="myocardial ischemia",
        parents=("MONDO:0005053", "MONDO:0024643"),
        xrefs=("SCTID:414545008",),
        matches=(("skos:exactMatch", "SCTID:414545008"),),
        obsolete=False,
        replaced_by=(),
    )


def test_version_is_the_header_verbatim(tmp_path):
    mondo = parse_obo(write_obo(tmp_path, MONDO_TEXT, "mondo.obo"), name="mondo")
    hpo = parse_obo(write_obo(tmp_path, HPO_TEXT, "hp.obo"), name="hpo")
    assert mondo.version == "releases/2026-10-06"
    assert hpo.version == "hp/releases/2026-09-01"


def test_obsolete_term_is_kept_with_its_replacement(tmp_path):
    term = parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo").terms["MONDO:9999998"]
    assert term.obsolete
    assert term.replaced_by == ("MONDO:0024644",)


def test_dangling_replacement_is_kept_as_an_id(tmp_path):
    term = parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo").terms["MONDO:9999999"]
    assert term.replaced_by == ("ENVO:03000144",)


def test_non_ascii_name_round_trips(tmp_path):
    term = parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo").terms["MONDO:9999997"]
    assert term.name == "Sjögren-like test term"


def test_parse_obo_hands_pronto_a_path_in_utf8(tmp_path, monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real = obo.pronto.Ontology

    def recording(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(obo.pronto, "Ontology", recording)
    parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo")
    [(args, kwargs)] = calls
    assert isinstance(args[0], pathlib.Path)
    assert kwargs == {"import_depth": 0, "encoding": "utf-8"}


def test_terms_are_sorted_and_reload_equal(tmp_path):
    path = write_obo(tmp_path, MONDO_TEXT)
    first = parse_obo(path, name="mondo")
    second = parse_obo(path, name="mondo")
    assert list(first.terms) == sorted(first.terms)
    assert dict(first.terms) == dict(second.terms)


def test_import_header_is_refused_offline(tmp_path, monkeypatch):
    _offline(monkeypatch)
    text = HEADER + "import: http://example.invalid/x.obo\n\n[Term]\nid: X:1\nname: one\n"
    assert "imports" in _refused(tmp_path, text)


def test_missing_file_is_unavailable(tmp_path, monkeypatch):
    _offline(monkeypatch)
    with pytest.raises(UnavailableError):
        parse_obo(tmp_path / "absent.obo", name="test")


def test_unhandled_construct_is_refused(tmp_path):
    text = (
        HEADER
        + "\n[Term]\nid: MONDO:0006715\nname: coronary stenosis\n"
        + "\n[Instance]\nid: X:1\ninstance_of: MONDO:0006715\n"
    )
    assert "skipped" in _refused(tmp_path, text)


def test_unread_date_is_logged(tmp_path, caplog):
    text = HEADER + "\n[Term]\nid: X:1\nname: one\ncreation_date: 2020-01-01\n"
    with caplog.at_level(logging.WARNING, logger="bko.ontology.obo"):
        ontology = parse_obo(write_obo(tmp_path, text), name="test")
    assert "X:1" in ontology.terms
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unread values" in warnings[0].getMessage()


def test_missing_data_version_is_refused(tmp_path):
    text = "format-version: 1.2\nontology: test\n\n[Term]\nid: X:1\nname: one\n"
    assert "data-version" in _refused(tmp_path, text)


def test_header_only_is_refused(tmp_path):
    assert "no term" in _refused(tmp_path, HEADER)


def test_not_obo_is_refused(tmp_path):
    assert "could not find a parser" in _refused(tmp_path, "hello\n")


def test_dangling_is_a_is_refused(tmp_path):
    text = HEADER + "\n[Term]\nid: X:1\nname: one\nis_a: MONDO:0000000\n"
    assert "KeyError: 'MONDO:0000000'" in _refused(tmp_path, text)


def test_single_intersection_of_is_refused(tmp_path):
    text = (
        HEADER
        + "\n[Term]\nid: X:2\nname: two\n"
        + "\n[Term]\nid: X:1\nname: one\nintersection_of: X:2\n"
    )
    assert "intersection_of" in _refused(tmp_path, text)


def test_load_ontology_verifies_first(tmp_path):
    data = MONDO_TEXT.encode()
    source = pin(data, name="mondo")
    plant(tmp_path / "good", source, data)
    assert load_ontology(source, tmp_path / "good").version == "releases/2026-10-06"
    # Still valid OBO, so only the pin can refuse it.
    changed = data.replace(b"name: coronary stenosis", b"name: coronary stenosiz")
    plant(tmp_path / "bad", source, changed)
    with pytest.raises(ContractError, match="pinned") as caught:
        load_ontology(source, tmp_path / "bad")
    assert not isinstance(caught.value, ReleaseFormatError)


def test_multi_valued_clauses_are_sorted_and_parents_are_direct(tmp_path):
    text = (
        HEADER
        + "idspace: skos http://www.w3.org/2004/02/skos/core#\n"
        + "\n[Term]\nid: X:9\nname: grandparent\n"
        + "\n[Term]\nid: X:8\nname: parent b\nis_a: X:9\n"
        + "\n[Term]\nid: X:7\nname: parent a\nis_a: X:9\n"
        + "\n[Term]\nid: X:1\nname: child\nis_a: X:8\nis_a: X:7\n"
        + "xref: Z:2\nxref: A:1\n"
        + "property_value: skos:narrowMatch Z:9\n"
        + "property_value: skos:exactMatch A:9\n"
        + "property_value: seeAlso Q:1\n"
        + 'property_value: skos:note "literal" xsd:string\n'
        + "\n[Term]\nid: X:2\nname: old\nis_obsolete: true\n"
        + "replaced_by: X:8\nreplaced_by: X:7\n"
    )
    terms = parse_obo(write_obo(tmp_path, text), name="test").terms
    assert terms["X:1"].parents == ("X:7", "X:8")
    assert terms["X:1"].xrefs == ("A:1", "Z:2")
    assert terms["X:1"].matches == (("skos:exactMatch", "A:9"), ("skos:narrowMatch", "Z:9"))
    assert terms["X:2"].replaced_by == ("X:7", "X:8")


def test_a_warning_from_elsewhere_is_logged(tmp_path, monkeypatch, caplog):
    real = obo.pronto.Ontology

    def warning(*args: Any, **kwargs: Any) -> object:
        warnings.warn("old API", DeprecationWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(obo.pronto, "Ontology", warning)
    with caplog.at_level(logging.WARNING, logger="bko.ontology.obo"):
        parse_obo(write_obo(tmp_path, MONDO_TEXT), name="mondo")
    assert "DeprecationWarning: old API" in caplog.text
    assert "unread values" not in caplog.text
