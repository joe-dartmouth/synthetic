"""The ICD-9 and Diabetes lab rows: their invariants, and `bko.ontology.check` on them."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from bko.ontology.check import Verdict, check
from bko.ontology.icd9 import (
    ICD9_MAPPINGS,
    LAB_MAPPINGS,
    UNMAPPED_ICD9,
    Icd9Mapping,
    LabMapping,
    UnmappedIcd9,
)
from bko.ontology.mapping import CHECKED_ROWS, MAPPED
from bko.ontology.obo import parse_obo
from bko.ontology.schema import CheckedRow
from bko.ontology.terms import Term
from tests._ontology_fixtures import write_obo
from tests.test_ontology_check import (
    CODE_ROW,
    ISCHEMIA,
    _CountingTerms,
    _mondo,
    _ontology,
    _problem,
    _term,
)


MEASURED = {
    "ICD9:250.4": ("MONDO:0005016", {"MONDO:0005016"}),
    "ICD9:401": ("MONDO:0001134", {"MONDO:0001134"}),
    "ICD9:410": ("MONDO:0004781", {"MONDO:0004781"}),
    "ICD9:428": ("MONDO:0005252", {"MONDO:0005009", "MONDO:0005252"}),
    "ICD9:434": ("MONDO:0000944", {"MONDO:0000944"}),
    "ICD9:486": ("MONDO:0005249", {"MONDO:0005249"}),
    "ICD9:491": ("MONDO:0005607", {"MONDO:0003781", "MONDO:0005607"}),
    "ICD9:496": ("MONDO:0005002", {"MONDO:0005002"}),
    "ICD9:585": ("MONDO:0005300", {"MONDO:0005300"}),
    "ICD9:707": ("MONDO:0004605", {"MONDO:0004605"}),
}
"""Each row's concept and the live holders of its xref in MONDO release 2026-10-06."""

LAB_STATES = {
    "A1Cresult": frozenset({"None", "Norm", ">7", ">8"}),
    "max_glu_serum": frozenset({"None", "Norm", ">200", ">300"}),
}
"""The result strings the Diabetes 130 archive writes in its two lab columns."""


def _xrefs(term: Term, *xrefs: str) -> Term:
    return replace(term, xrefs=tuple(sorted(xrefs)))


COPD = _xrefs(
    _term("MONDO:0005002", "chronic obstructive pulmonary disease"), "ICD9:490-496", "ICD9:496"
)
COPD_ROW = Icd9Mapping(
    codes=("496",),
    xref="ICD9:496",
    concept="MONDO:0005002",
    label="chronic obstructive pulmonary disease",
    holders=frozenset({"MONDO:0005002"}),
    source="mondo",
    limits="test",
)
HEART_FAILURE = _xrefs(_term("MONDO:0005252", "heart failure"), "ICD9:428", "ICD9:428.9")
CONGESTIVE = _xrefs(_term("MONDO:0005009", "congestive heart failure"), "ICD9:428", "ICD9:428.0")
HEART_FAILURE_ROW = Icd9Mapping(
    codes=("428",),
    xref="ICD9:428",
    concept="MONDO:0005252",
    label="heart failure",
    holders=frozenset({"MONDO:0005009", "MONDO:0005252"}),
    source="mondo",
    limits="test",
)
CHRONIC_BRONCHITIS = _xrefs(_term("MONDO:0005607", "chronic bronchitis"), "ICD9:491")
BRONCHITIS = _xrefs(_term("MONDO:0003781", "bronchitis"), "ICD9:491")
BRONCHITIS_ROW = Icd9Mapping(
    codes=("491",),
    xref="ICD9:491",
    concept="MONDO:0005607",
    label="chronic bronchitis",
    holders=frozenset({"MONDO:0003781", "MONDO:0005607"}),
    source="mondo",
    limits="test",
)
EXCLUDED_ROW = UnmappedIcd9(
    xref="ICD9:414",
    label="Other forms of chronic ischemic heart disease",
    reason="test",
    source="mondo",
)
A1C_ROW = LabMapping(
    field="A1Cresult",
    states=frozenset({">8", ">7"}),
    concept="HP:0040217",
    label="Elevated hemoglobin A1c",
    source="strack-2014",
    limits="test",
)


def test_the_rows_hold_the_measured_concepts_and_holders() -> None:
    assert {row.xref: (row.concept, set(row.holders)) for row in ICD9_MAPPINGS} == MEASURED
    for row in ICD9_MAPPINGS:
        assert row.concept in row.holders, row
        assert row.xref in {f"ICD9:{code}" for code in row.codes}, row
        for code in row.codes:
            assert re.fullmatch(r"[0-9]{3}(\.[0-9]{1,2})?", code, re.ASCII), code


def test_excluded_categories_are_disjoint_from_mapped_ones() -> None:
    excluded = [row.xref for row in UNMAPPED_ICD9]
    assert excluded
    assert len(excluded) == len(set(excluded))
    assert {row.xref for row in ICD9_MAPPINGS}.isdisjoint(excluded)


def test_lab_rows_name_the_archives_states() -> None:
    assert LAB_MAPPINGS
    for row in LAB_MAPPINGS:
        assert row.states, row
        assert row.states <= LAB_STATES[row.field] - {"None", "Norm"}, row
        assert row.concept.startswith("HP:"), row


def test_the_diabetes_rows_join_the_checked_table() -> None:
    checked = set(CHECKED_ROWS)
    assert set(ICD9_MAPPINGS) <= checked
    assert set(UNMAPPED_ICD9) <= checked
    assert set(LAB_MAPPINGS) <= checked
    assert {row.concept for row in (*ICD9_MAPPINGS, *LAB_MAPPINGS)} <= MAPPED


def test_an_icd9_row_passes_when_its_concept_carries_the_exact_xref() -> None:
    assert check((COPD_ROW,), {"MONDO": _mondo(COPD)}) == (
        Verdict("icd9", "ICD9:496", "MONDO:0005002", "chronic obstructive pulmonary disease", None),
    )


@pytest.mark.parametrize(
    "xref", ["ICD9:496.0", "ICD9:4960", "ICD9CM:496", "ICD10CM:496", "ICD9:490-496", "ICD9:96"]
)
def test_a_missing_xref_is_reported_naming_the_release(xref: str) -> None:
    near = _xrefs(_term("MONDO:0005002", "chronic obstructive pulmonary disease"), xref)
    assert _problem(COPD_ROW, {"MONDO": _mondo(near)}) == (
        "mondo releases/2026-10-06 has no xref ICD9:496 on MONDO:0005002"
    )


def test_the_declared_holders_must_equal_the_live_holders() -> None:
    both = {"MONDO": _mondo(HEART_FAILURE, CONGESTIVE, CHRONIC_BRONCHITIS, BRONCHITIS)}
    verdicts = check((HEART_FAILURE_ROW, BRONCHITIS_ROW), both)
    assert [v.problem for v in verdicts] == [None, None]
    assert _problem(HEART_FAILURE_ROW, {"MONDO": _mondo(HEART_FAILURE)}) == (
        "xref ICD9:428 is on MONDO:0005252; the row declares MONDO:0005009, MONDO:0005252"
    )
    third = _xrefs(_term("MONDO:0000003", "third"), "ICD9:491")
    assert _problem(BRONCHITIS_ROW, {"MONDO": _mondo(CHRONIC_BRONCHITIS, BRONCHITIS, third)}) == (
        "xref ICD9:491 is on MONDO:0000003, MONDO:0003781, MONDO:0005607; "
        "the row declares MONDO:0003781, MONDO:0005607"
    )


QUALIFIED_MONDO = """\
format-version: 1.2
data-version: releases/2026-10-06
ontology: mondo

[Term]
id: MONDO:0001134
name: essential hypertension
xref: ICD9:401 {source="DOID:10825"}

[Term]
id: MONDO:0005249
name: pneumonia
xref: ICD9:486 {source="MONDO:equivalentTo", source="MONDO:i2s"}

[Term]
id: MONDO:0005002
name: chronic obstructive pulmonary disease
xref: ICD9:496 {source="EFO:0000341", source="MONDO:relatedTo", source="MONDO:i2s"}
"""
"""Three stanzas abridged from MONDO release 2026-10-06 to the id, name and ICD-9 xref, each xref
with one of the release's qualifier shapes."""


def test_any_xref_qualifier_satisfies_the_check(tmp_path: Path) -> None:
    mondo = parse_obo(write_obo(tmp_path, QUALIFIED_MONDO, "mondo.obo"), name="mondo")
    rows = [row for row in ICD9_MAPPINGS if row.xref in {"ICD9:401", "ICD9:486", "ICD9:496"}]
    assert len(rows) == 3
    assert [v.problem for v in check(rows, {"MONDO": mondo})] == [None, None, None]


def test_an_unmapped_icd9_code_that_gains_an_exact_holder_is_reported() -> None:
    assert check((EXCLUDED_ROW,), {"MONDO": _mondo(ISCHEMIA)}) == (
        Verdict("excluded-icd9", "ICD9:414", "", EXCLUDED_ROW.label, None),
    )
    gained = _xrefs(_term("MONDO:0000002", "gained"), "ICD9:414")
    assert _problem(EXCLUDED_ROW, {"MONDO": _mondo(ISCHEMIA, gained)}) == (
        "xref ICD9:414 is on MONDO:0000002"
    )
    subcode = _xrefs(_term("MONDO:0000003", "subcode"), "ICD9:414.0")
    [verdict] = check((EXCLUDED_ROW,), {"MONDO": _mondo(subcode)})
    assert verdict.problem is None


def test_an_unmapped_icd9_code_without_mondo_is_a_verdict() -> None:
    hp_only = {"HP": _ontology("hpo", "hp/releases/2026-09-01")}
    assert _problem(EXCLUDED_ROW, hp_only) == "no release for prefix MONDO"


def test_an_xref_on_an_obsolete_holder_is_ignored() -> None:
    gone = replace(CONGESTIVE, obsolete=True)
    row = replace(HEART_FAILURE_ROW, holders=frozenset({"MONDO:0005252"}))
    [verdict] = check((row,), {"MONDO": _mondo(HEART_FAILURE, gone)})
    assert verdict.problem is None
    gone_holder = replace(_xrefs(_term("MONDO:0000002", "gone"), "ICD9:414"), obsolete=True)
    [verdict] = check((EXCLUDED_ROW,), {"MONDO": _mondo(gone_holder)})
    assert verdict.problem is None


def test_icd9_and_snomed_rows_share_one_scan() -> None:
    def scans(*rows: CheckedRow) -> int:
        terms = _CountingTerms(_mondo(ISCHEMIA, COPD).terms)
        check(rows, {"MONDO": replace(_mondo(), terms=terms)})
        return terms.scans

    assert scans(COPD_ROW) > 0, "the control: checking an ICD-9 row reads the release"
    assert scans(CODE_ROW, COPD_ROW, EXCLUDED_ROW) == scans(CODE_ROW)


def test_a_lab_row_is_a_field_row_named_by_its_states() -> None:
    hpo = _ontology("hpo", "hp/releases/2026-09-01", _term("HP:0040217", "Elevated hemoglobin A1c"))
    assert check((A1C_ROW,), {"HP": hpo}) == (
        Verdict("field", "A1Cresult in >7|>8", "HP:0040217", "Elevated hemoglobin A1c", None),
    )


def test_an_icd9_row_checks_its_concept_like_any_concept_row() -> None:
    renamed = _xrefs(_term("MONDO:0005002", "COPD"), "ICD9:496")
    assert _problem(COPD_ROW, {"MONDO": _mondo(renamed)}) == (
        "named 'COPD' in mondo releases/2026-10-06"
    )


def test_an_icd9_rows_xref_is_its_first_category() -> None:
    assert all(row.xref == f"ICD9:{row.codes[0]}" for row in ICD9_MAPPINGS)
    with pytest.raises(ValueError, match=re.escape("xref ICD9:496 is not ICD9:491")):
        replace(COPD_ROW, codes=("491",))
    with pytest.raises(ValueError, match="no ICD-9 categories"):
        replace(COPD_ROW, codes=())


def test_a_side_with_no_holder_is_rendered_none() -> None:
    bare = replace(COPD_ROW, holders=frozenset())
    assert _problem(bare, {"MONDO": _mondo(COPD)}) == (
        "xref ICD9:496 is on MONDO:0005002; the row declares none"
    )
