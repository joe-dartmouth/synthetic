"""`bko.ontology.obo` keeps the literal annotations of `ANNOTATION_PREDICATES` and nothing else."""

from __future__ import annotations

import logging

import pytest

from bko.ontology.obo import ANNOTATION_PREDICATES, parse_obo
from bko.ontology.terms import SOC_ID
from tests._occupation_fixtures import CHIEF_EXECUTIVE, OCCO_TEXT
from tests._ontology_fixtures import HPO_TEXT, MONDO_TEXT, write_obo


pytestmark = pytest.mark.filterwarnings("error")


def test_a_literal_annotation_of_a_kept_predicate_is_read(tmp_path):
    occo = parse_obo(write_obo(tmp_path, OCCO_TEXT), name="occo")
    chief = occo.terms["OCCO:11101100"]
    assert ANNOTATION_PREDICATES == ("OCCO:00000008",)
    assert chief.annotations == (("OCCO:00000008", "11-1011.00"),)
    assert chief.annotation(SOC_ID) == ("11-1011.00",)
    assert chief.annotation("IAO:0000117") == ()
    assert occo.terms["OCCO:11303100"].annotation(SOC_ID) == ("11-3031",)
    assert occo.terms["OCCO:00000001"].annotations == ()


def test_two_literals_of_a_kept_predicate_are_both_kept_sorted(tmp_path):
    stanza = CHIEF_EXECUTIVE + 'property_value: OCCO:00000008 "11-1011.03" xsd:string\n'
    occo = parse_obo(write_obo(tmp_path, OCCO_TEXT.replace(CHIEF_EXECUTIVE, stanza)), name="occo")
    assert occo.terms["OCCO:11101100"].annotation(SOC_ID) == ("11-1011.00", "11-1011.03")


def test_a_resource_value_of_a_kept_predicate_is_not_a_literal_and_is_counted(tmp_path, caplog):
    stanza = CHIEF_EXECUTIVE.replace(
        'property_value: OCCO:00000008 "11-1011.00" xsd:string\n',
        "property_value: OCCO:00000008 OCCO:11101000\n",
    )
    path = write_obo(tmp_path, OCCO_TEXT.replace(CHIEF_EXECUTIVE, stanza))
    with caplog.at_level(logging.WARNING, logger="bko.ontology.obo"):
        occo = parse_obo(path, name="occo")
    assert occo.terms["OCCO:11101100"].annotations == ()
    assert caplog.messages == [
        f"{path}: 1 property_values of OCCO:00000008 are not literals, which no term keeps"
    ]


def test_a_release_whose_kept_values_are_all_literals_logs_no_count(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="bko.ontology.obo"):
        parse_obo(write_obo(tmp_path, OCCO_TEXT), name="occo")
    assert caplog.messages == []


@pytest.mark.parametrize(("text", "name"), [(MONDO_TEXT, "mondo"), (HPO_TEXT, "hpo")])
def test_terms_of_releases_without_it_are_unchanged(tmp_path, text, name):
    ontology = parse_obo(write_obo(tmp_path, text), name=name)
    assert ontology.terms
    assert all(term.annotations == () for term in ontology.terms.values())
