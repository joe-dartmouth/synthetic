"""`bko.ontology.check` against hand-built releases; no pronto."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from bko.ontology.check import Verdict, check
from bko.ontology.schema import AtLeast, CodeMapping, Codes, FieldMapping, OutOfScope
from bko.ontology.terms import Ontology, Term


MATCH = ("skos:exactMatch", "SCTID:414545008")


def _term(
    curie: str,
    name: str,
    *,
    matches: tuple[tuple[str, str], ...] = (),
    obsolete: bool = False,
    replaced_by: tuple[str, ...] = (),
) -> Term:
    return Term(
        curie=curie,
        name=name,
        parents=(),
        xrefs=(),
        matches=matches,
        obsolete=obsolete,
        replaced_by=replaced_by,
    )


def _ontology(name: str, version: str, *terms: Term) -> Ontology:
    return Ontology(
        name=name,
        version=version,
        terms=MappingProxyType({term.curie: term for term in sorted(terms, key=lambda t: t.curie)}),
    )


def _mondo(*terms: Term) -> Ontology:
    return _ontology("mondo", "releases/2026-10-06", *terms)


ISCHEMIA = _term("MONDO:0024644", "myocardial ischemia", matches=(MATCH,))
STENOSIS = _term("MONDO:0006715", "coronary stenosis")
DIABETES = _term("MONDO:0005015", "diabetes mellitus")
DEPRESSION = _term("HP:0012250", "ST segment depression")

LABEL_ROW = FieldMapping(
    when=(Codes("target", frozenset({1})),),
    concept="MONDO:0006715",
    label="coronary stenosis",
    source="uci-heart-disease",
    limits="test",
)
ST_ROW = FieldMapping(
    when=(AtLeast("oldpeak", 1.0), Codes("slope", frozenset({2, 3}))),
    concept="HP:0012250",
    label="ST segment depression",
    source="gibbons-2002",
    limits="test",
)
CODE_ROW = CodeMapping(
    snomed="414545008", concept="MONDO:0024644", label="myocardial ischemia", source="mondo"
)
SCOPE_ROW = OutOfScope(
    concept="MONDO:0005015", label="diabetes mellitus", reason="test", source="ada-2026"
)


def _agreeing() -> dict[str, Ontology]:
    return {
        "MONDO": _mondo(ISCHEMIA, STENOSIS, DIABETES),
        "HP": _ontology("hpo", "hp/releases/2026-09-01", DEPRESSION),
    }


def test_agreeing_rows_pass_in_input_order() -> None:
    verdicts = check((LABEL_ROW, ST_ROW, CODE_ROW, SCOPE_ROW), _agreeing())
    assert verdicts == (
        Verdict("field", "target in 1", "MONDO:0006715", "coronary stenosis", None),
        Verdict(
            "field", "oldpeak >= 1 and slope in 2|3", "HP:0012250", "ST segment depression", None
        ),
        Verdict("code", "SCTID:414545008", "MONDO:0024644", "myocardial ischemia", None),
        Verdict("out-of-scope", "", "MONDO:0005015", "diabetes mellitus", None),
    )


def test_no_rows_give_no_verdicts() -> None:
    assert check((), _agreeing()) == ()


def _problem(row: FieldMapping | CodeMapping | OutOfScope, ontologies: dict[str, Ontology]) -> str:
    [verdict] = check((row,), ontologies)
    assert verdict.problem is not None
    return verdict.problem


def test_unknown_prefix() -> None:
    row = OutOfScope(concept="DOID:0050000", label="x", reason="test", source="mondo")
    assert _problem(row, _agreeing()) == "no release for prefix DOID"


def test_absent_concept() -> None:
    assert _problem(LABEL_ROW, {"MONDO": _mondo(ISCHEMIA)}) == (
        "absent from mondo releases/2026-10-06"
    )


def test_obsolete_concept_names_its_replacement() -> None:
    obsolete = _term(
        "MONDO:0006715", "coronary stenosis", obsolete=True, replaced_by=("MONDO:0024644",)
    )
    assert _problem(LABEL_ROW, {"MONDO": _mondo(obsolete)}) == (
        "obsolete in mondo releases/2026-10-06; replaced by MONDO:0024644"
    )


def test_obsolete_concept_without_replacement() -> None:
    obsolete = _term("MONDO:0006715", "coronary stenosis", obsolete=True)
    assert _problem(LABEL_ROW, {"MONDO": _mondo(obsolete)}) == (
        "obsolete in mondo releases/2026-10-06; replaced by nothing"
    )


def test_label_is_compared_case_sensitively() -> None:
    renamed = _term("MONDO:0024644", "Myocardial Ischemia", matches=(MATCH,))
    assert _problem(CODE_ROW, {"MONDO": _mondo(renamed)}) == (
        "named 'Myocardial Ischemia' in mondo releases/2026-10-06"
    )


def test_code_row_needs_the_exact_match() -> None:
    bare = _term("MONDO:0024644", "myocardial ischemia")
    assert _problem(CODE_ROW, {"MONDO": _mondo(bare)}) == (
        "mondo releases/2026-10-06 has no skos:exactMatch SCTID:414545008 on MONDO:0024644"
    )


def test_code_row_refuses_a_second_live_holder() -> None:
    other = _term("MONDO:0000002", "other", matches=(MATCH,))
    assert _problem(CODE_ROW, {"MONDO": _mondo(ISCHEMIA, other)}) == (
        "skos:exactMatch SCTID:414545008 is also on MONDO:0000002"
    )


def test_an_obsolete_holder_is_ignored() -> None:
    other = _term("MONDO:0000002", "other", matches=(MATCH,), obsolete=True)
    [verdict] = check((CODE_ROW,), {"MONDO": _mondo(ISCHEMIA, other)})
    assert verdict.problem is None


@pytest.mark.parametrize(
    "match",
    [
        ("skos:exactMatch", "SCTID:4145450080"),
        ("skos:exactMatch", "SCTID:14145450080"),
        ("skos:exactMatch", "OMIM:414545008"),
        ("skos:exactMatch", "SCTID:414545008 "),
        ("skos:closeMatch", "SCTID:414545008"),
    ],
)
def test_near_miss_matches_back_nothing(match: tuple[str, str]) -> None:
    near = _term("MONDO:0024644", "myocardial ischemia", matches=(match,))
    assert _problem(CODE_ROW, {"MONDO": _mondo(near)}).startswith(
        "mondo releases/2026-10-06 has no"
    )


def test_hp_row_is_looked_up_only_in_hp() -> None:
    mondo_with_hp_id = _mondo(_term("HP:0012250", "ST segment depression"))
    assert _problem(ST_ROW, {"MONDO": mondo_with_hp_id}) == "no release for prefix HP"
