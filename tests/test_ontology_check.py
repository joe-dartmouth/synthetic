"""`bko.ontology.check` against hand-built releases, and the table against the fixture releases."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from bko.ontology.check import Verdict, check
from bko.ontology.mapping import CHECKED_ROWS, CODE_MAPPINGS, MAPPED, OUT_OF_SCOPE
from bko.ontology.obo import parse_obo
from bko.ontology.schema import (
    AtLeast,
    CheckedRow,
    CodedRow,
    CodeMapping,
    Codes,
    FieldMapping,
    OutOfScope,
    UnmappedCode,
)
from bko.ontology.terms import Ontology, Term, ancestors
from tests._ontology_fixtures import HPO_LABELS, HPO_TEXT, MONDO_TEXT, write_obo


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
EXCLUDED_ROW = UnmappedCode(
    snomed="401314000",
    label="Acute non-ST segment elevation myocardial infarction (disorder)",
    reason="test",
    source="mondo",
)
EXCLUDED_MATCH = ("skos:exactMatch", "SCTID:401314000")


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


def _problem(row: CheckedRow, ontologies: dict[str, Ontology]) -> str:
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


def test_a_nameless_concept_is_reported() -> None:
    nameless = replace(_term("MONDO:0006715", "x"), name=None)
    assert _problem(LABEL_ROW, {"MONDO": _mondo(nameless)}) == (
        "has no name in mondo releases/2026-10-06"
    )


def _child_of(curie: str, name: str, parent: str) -> Term:
    return replace(_term(curie, name), parents=(parent,))


def test_an_ancestor_of_a_mapped_concept_is_in_scope() -> None:
    stenosis = _child_of("MONDO:0006715", "coronary stenosis", "MONDO:0000001")
    middle = _child_of("MONDO:0000001", "middle", "MONDO:0005015")
    ontologies = {"MONDO": _mondo(stenosis, middle, DIABETES)}
    verdicts = check((LABEL_ROW, SCOPE_ROW), ontologies)
    assert verdicts[0].problem is None
    assert verdicts[1].problem == (
        "is_a ancestor of mapped MONDO:0006715 in mondo releases/2026-10-06, so it is in scope"
    )


def test_an_ancestor_reached_only_through_another_prefix_is_out_of_scope() -> None:
    stenosis = _child_of("MONDO:0006715", "coronary stenosis", "BFO:0000016")
    bridge = _child_of("BFO:0000016", "disposition", "MONDO:0005015")
    verdicts = check((LABEL_ROW, SCOPE_ROW), {"MONDO": _mondo(stenosis, bridge, DIABETES)})
    assert [v.problem for v in verdicts] == [None, None]


def test_an_unrelated_out_of_scope_concept_passes_without_mapped_rows() -> None:
    stenosis = _child_of("MONDO:0006715", "coronary stenosis", "MONDO:0005015")
    [verdict] = check((SCOPE_ROW,), {"MONDO": _mondo(stenosis, DIABETES)})
    assert verdict.problem is None


def _fixture_releases(tmp_path: Path) -> dict[str, Ontology]:
    return {
        "MONDO": parse_obo(write_obo(tmp_path, MONDO_TEXT, "mondo.obo"), name="mondo"),
        "HP": parse_obo(write_obo(tmp_path, HPO_TEXT, "hp.obo"), name="hpo"),
    }


def test_the_table_agrees_with_the_fixture_releases(tmp_path: Path) -> None:
    verdicts = check(CHECKED_ROWS, _fixture_releases(tmp_path))
    assert [v.problem for v in verdicts] == [None] * len(CHECKED_ROWS)
    assert "excluded-code" in {v.table for v in verdicts}
    table_hp = {
        row.concept: row.label
        for row in CHECKED_ROWS
        if not isinstance(row, UnmappedCode) and row.concept.startswith("HP:")
    }
    assert table_hp == dict(HPO_LABELS)


def test_no_out_of_scope_row_is_an_ancestor_of_a_mapped_concept(tmp_path: Path) -> None:
    mondo = _fixture_releases(tmp_path)["MONDO"]
    above = {
        ancestor
        for concept in MAPPED
        if concept.startswith("MONDO:")
        for ancestor in ancestors(mondo, concept, prefix="MONDO")
    }
    assert "MONDO:0005015" in above, "the control: type 2 diabetes reaches diabetes mellitus"
    assert above.isdisjoint(row.concept for row in OUT_OF_SCOPE)


def test_diabetes_mellitus_is_in_scope_once_type_2_diabetes_maps(tmp_path: Path) -> None:
    [diabetes] = [row for row in CODE_MAPPINGS if row.label == "type 2 diabetes mellitus"]
    verdicts = check((diabetes, SCOPE_ROW), _fixture_releases(tmp_path))
    assert [v.problem for v in verdicts] == [
        None,
        "is_a ancestor of mapped MONDO:0005148 in mondo releases/2026-10-06, so it is in scope",
    ]


def test_an_excluded_code_without_a_holder_passes() -> None:
    assert check((EXCLUDED_ROW,), _agreeing()) == (
        Verdict("excluded-code", "SCTID:401314000", "", EXCLUDED_ROW.label, None),
    )


def test_an_excluded_code_that_gains_a_live_exact_match_is_reported() -> None:
    gained = _term("MONDO:0000002", "gained", matches=(EXCLUDED_MATCH,))
    assert _problem(EXCLUDED_ROW, {"MONDO": _mondo(ISCHEMIA, gained)}) == (
        "skos:exactMatch SCTID:401314000 is on MONDO:0000002"
    )


def test_an_excluded_code_ignores_obsolete_and_close_holders() -> None:
    gone = _term("MONDO:0000002", "gone", matches=(EXCLUDED_MATCH,), obsolete=True)
    close = _term("MONDO:0000003", "close", matches=(("skos:closeMatch", "SCTID:401314000"),))
    [verdict] = check((EXCLUDED_ROW,), {"MONDO": _mondo(gone, close)})
    assert verdict.problem is None


def test_an_excluded_code_without_mondo_is_a_verdict() -> None:
    hp_only = {"HP": _ontology("hpo", "hp/releases/2026-09-01", DEPRESSION)}
    assert _problem(EXCLUDED_ROW, hp_only) == "no release for prefix MONDO"


class _CountingTerms(Mapping[str, Term]):
    """A term table that counts its scans: each iteration, which `items()` also makes."""

    def __init__(self, terms: Mapping[str, Term]) -> None:
        self._terms = terms
        self.scans = 0

    def __getitem__(self, curie: str) -> Term:
        return self._terms[curie]

    def __iter__(self) -> Iterator[str]:
        self.scans += 1
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)


SECOND = _term("MONDO:0000002", "second", matches=(("skos:exactMatch", "SCTID:2"),))
SECOND_ROW = CodeMapping(snomed="2", concept="MONDO:0000002", label="second", source="mondo")


def _scans(*rows: CodedRow) -> int:
    terms = _CountingTerms(_mondo(ISCHEMIA, SECOND).terms)
    check(rows, {"MONDO": replace(_mondo(), terms=terms)})
    return terms.scans


def test_a_second_code_row_does_not_scan_the_release_again() -> None:
    assert _scans(CODE_ROW) > 0, "the control: checking a code row reads the release"
    assert _scans(CODE_ROW, SECOND_ROW) == _scans(CODE_ROW)


def test_an_excluded_code_shares_the_code_rows_scan() -> None:
    assert _scans(CODE_ROW, EXCLUDED_ROW) == _scans(CODE_ROW)


def test_a_code_row_without_its_release_is_a_verdict() -> None:
    assert _problem(replace(CODE_ROW, concept="DOID:0000001"), _agreeing()) == (
        "no release for prefix DOID"
    )


def test_holders_are_listed_in_term_order_and_obsolete_ones_are_left_out() -> None:
    late = _term("MONDO:0000003", "late", matches=(MATCH,))
    early = _term("MONDO:0000002", "early", matches=(MATCH,))
    gone = _term("MONDO:0000004", "gone", matches=(MATCH,), obsolete=True)
    assert _problem(CODE_ROW, {"MONDO": _mondo(ISCHEMIA, late, early, gone)}) == (
        "skos:exactMatch SCTID:414545008 is also on MONDO:0000002, MONDO:0000003"
    )


def test_a_holder_naming_the_match_twice_is_listed_once() -> None:
    twice = _term("MONDO:0000002", "other", matches=(MATCH, MATCH))
    assert _problem(CODE_ROW, {"MONDO": _mondo(ISCHEMIA, twice)}) == (
        "skos:exactMatch SCTID:414545008 is also on MONDO:0000002"
    )


def test_a_close_match_on_another_term_is_not_a_holder() -> None:
    close = _term("MONDO:0000002", "other", matches=(("skos:closeMatch", "SCTID:414545008"),))
    [verdict] = check((CODE_ROW,), {"MONDO": _mondo(ISCHEMIA, close)})
    assert verdict.problem is None
