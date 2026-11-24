"""`bko.ontology.codes`: the generated ICD-9 code rows, their rule, their text and their command."""

from __future__ import annotations

import io
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from bko.datasets.diagnoses import code_number
from bko.ontology.__main__ import Context, main
from bko.ontology.check import check
from bko.ontology.codes import REASONS, Generated, generate, render
from bko.ontology.icd9 import EXACT, ICD9_MAPPINGS, Icd9Code
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from bko.ontology.terms import Term
from tests._datasets_fixtures import pin
from tests._diabetes_fixtures import archive, diabetic_line, text
from tests._ontology_fixtures import RoutingOpener
from tests.test_ontology_check import _mondo, _term


def _holder(
    curie: str,
    name: str,
    *xrefs: str,
    parents: tuple[str, ...] = (),
    sources: tuple[str, ...] = (EXACT,),
) -> Term:
    """A term carrying each of `xrefs`, each qualified by `sources` (none when empty)."""
    qualifiers = tuple(sorted(("source", source) for source in sources))
    return replace(
        _term(curie, name),
        xrefs=tuple(sorted(xrefs)),
        parents=parents,
        xref_qualifiers=tuple((xref, qualifiers) for xref in sorted(xrefs)) if sources else (),
    )


HEART_FAILURE = _holder("MONDO:0005252", "heart failure", "ICD9:428")
CONGESTIVE = _holder(
    "MONDO:0005009", "congestive heart failure", "ICD9:428", parents=("MONDO:0005252",)
)
BRONCHITIS = _holder("MONDO:0003781", "bronchitis", "ICD9:491")
CHRONIC = _holder("MONDO:0005607", "chronic bronchitis", "ICD9:491", parents=("MONDO:0003781",))


def test_the_curated_concept_wins_for_a_code_it_maps() -> None:
    [row] = generate(_mondo(BRONCHITIS, CHRONIC), ["491"]).rows
    assert row == Icd9Code(
        "491", "MONDO:0005607", "chronic bronchitis", frozenset({"MONDO:0003781", "MONDO:0005607"})
    )
    [uncurated] = generate(_mondo(BRONCHITIS, CHRONIC), ["491"], curated=()).rows
    assert uncurated.concept == "MONDO:0003781"


def test_a_curated_concept_that_is_not_an_exact_holder_is_refused() -> None:
    weak = _holder("MONDO:0005607", "chronic bronchitis", "ICD9:491", sources=("DOID:6132",))
    [curated] = [row for row in ICD9_MAPPINGS if row.xref == "ICD9:491"]
    assert curated.concept == weak.curie
    with pytest.raises(ValueError, match="ICD9:491 maps MONDO:0005607, which is not among"):
        generate(_mondo(BRONCHITIS, weak), ["491"])


def test_the_holder_every_other_descends_from_is_mapped() -> None:
    middle = _holder("MONDO:0000002", "middle", parents=("MONDO:0000001",))
    top = _holder("MONDO:0000001", "intestinal infection", "ICD9:008")
    below = _holder("MONDO:0000003", "below", "ICD9:008", parents=("MONDO:0000002",))
    generated = generate(_mondo(top, middle, below), ["8", "008"], curated=())
    assert generated.rows == (
        Icd9Code(
            "008",
            "MONDO:0000001",
            "intestinal infection",
            frozenset({"MONDO:0000001", "MONDO:0000003"}),
        ),
    )
    assert sum(generated.unmapped.values()) == 0


def test_holders_that_are_not_nested_leave_the_code_unmapped() -> None:
    first = _holder("MONDO:0000001", "first", "ICD9:558")
    second = _holder("MONDO:0000002", "second", "ICD9:558")
    generated = generate(_mondo(first, second), ["558"])
    assert generated.rows == ()
    assert generated.unmapped["holders not nested"] == 1


def test_an_obsolete_holder_is_ignored() -> None:
    old = replace(_holder("MONDO:0000009", "obsolete x", "ICD9:428"), obsolete=True)
    only_old = replace(_holder("MONDO:0000008", "obsolete y", "ICD9:999"), obsolete=True)
    generated = generate(_mondo(HEART_FAILURE, CONGESTIVE, old, only_old), ["428", "999"])
    [row] = generated.rows
    assert row.holders == {"MONDO:0005009", "MONDO:0005252"}
    assert generated.unmapped["obsolete holder only"] == 1


def test_only_an_equivalent_to_xref_maps_a_code() -> None:
    exact = _holder("MONDO:0000001", "exact", "ICD9:100")
    related = _holder("MONDO:0000002", "related", "ICD9:100", sources=("MONDO:relatedTo",))
    generated = generate(_mondo(exact, related), ["100"], curated=())
    assert generated.rows == (Icd9Code("100", "MONDO:0000001", "exact", frozenset({exact.curie})),)


@pytest.mark.parametrize(
    ("sources", "reason"),
    [
        (("MONDO:relatedTo", "MONDO:i2s"), "relatedTo only"),
        (("MONDO:RelatedTo", "MONDO:i2s"), "relatedTo only"),
        (("DOID:10825",), "unreviewed source only"),
        (("EFO:0000401", "DOID:9351"), "unreviewed source only"),
        ((), "unreviewed source only"),
        (("MONDO:obsoleteEquivalent", "MONDO:i2s"), "unreviewed source only"),
    ],
)
def test_a_code_held_under_another_qualifier_is_unmapped_by_reason(
    sources: tuple[str, ...], reason: str
) -> None:
    holder = _holder("MONDO:0000001", "x", "ICD9:401", sources=sources)
    generated = generate(_mondo(holder), ["401"], curated=())
    assert generated.rows == ()
    assert {key: n for key, n in generated.unmapped.items() if n} == {reason: 1}


def test_a_related_holder_beside_an_unreviewed_one_counts_as_related_to() -> None:
    related = _holder("MONDO:0000001", "x", "ICD9:496", sources=("MONDO:relatedTo",))
    unreviewed = _holder("MONDO:0000002", "y", "ICD9:496", sources=("DOID:3083",))
    obsolete = replace(_holder("MONDO:0000003", "z", "ICD9:496"), obsolete=True)
    generated = generate(_mondo(unreviewed, related, obsolete), ["496"], curated=())
    assert {key: n for key, n in generated.unmapped.items() if n} == {"relatedTo only": 1}


def test_codes_without_a_holder_and_v_and_e_codes_are_counted_by_reason() -> None:
    v_holder = _holder("MONDO:0000001", "v holder", "ICD9:V45")
    generated = generate(_mondo(v_holder), ["?", "V45", "V45.81", "E888.1", "123.4"])
    assert generated.rows == ()
    assert tuple(generated.unmapped) == REASONS
    assert dict(generated.unmapped) == {
        "no live xref": 1,
        "obsolete holder only": 0,
        "relatedTo only": 0,
        "unreviewed source only": 0,
        "holders not nested": 0,
        "V code": 2,
        "E code": 1,
    }


def test_rows_follow_the_encoding_order_not_the_text_order() -> None:
    narrow = _holder("MONDO:0000001", "a", "ICD9:250.13")
    broad = _holder("MONDO:0000002", "b", "ICD9:250.2")
    generated = generate(_mondo(narrow, broad), ["250.13", "250.2"], curated=())
    assert [row.code for row in generated.rows] == ["250.2", "250.13"]
    assert code_number("250.2") < code_number("250.13")


def test_generated_compares_by_identity() -> None:
    generated = generate(_mondo(HEART_FAILURE), ["428"], curated=())
    same = generated
    assert same == generated
    assert generated != Generated(generated.rows, generated.unmapped)


def test_a_concept_without_a_name_is_refused() -> None:
    nameless = replace(_holder("MONDO:0000001", "x", "ICD9:123"), name=None)
    with pytest.raises(ValueError, match="MONDO:0000001 has no name"):
        generate(_mondo(nameless), ["123"])


def test_each_committed_code_is_one_row_of_a_mondo_concept() -> None:
    assert ICD9_CODES
    assert len({code_number(row.code) for row in ICD9_CODES}) == len(ICD9_CODES)
    for row in ICD9_CODES:
        assert row.concept.startswith("MONDO:"), row


def test_the_curated_rows_win_in_the_committed_table() -> None:
    committed = {row.xref: row.concept for row in ICD9_CODES}
    for curated in ICD9_MAPPINGS:
        assert committed.get(curated.xref, curated.concept) == curated.concept, curated


def test_the_generated_rows_join_the_checked_table() -> None:
    assert set(ICD9_CODES) <= set(CONCEPT_ROWS)
    assert set(ICD9_CODES) <= set(CHECKED_ROWS)


ERYSIPELAS = Icd9Code("035", "MONDO:0001266", "erysipelas", frozenset({"MONDO:0001266"}))


def test_a_code_row_agrees_with_an_exact_holder_and_ignores_weaker_ones() -> None:
    concept = _holder("MONDO:0001266", "erysipelas", "ICD9:035")
    weaker = _holder("MONDO:0000002", "y", "ICD9:035", sources=("MONDO:relatedTo",))
    [verdict] = check((ERYSIPELAS,), {"MONDO": _mondo(concept, weaker)})
    assert (verdict.table, verdict.subject, verdict.problem) == ("icd9-code", "ICD9:035", None)


def test_a_code_row_with_an_undeclared_exact_holder_disagrees() -> None:
    concept = _holder("MONDO:0001266", "erysipelas", "ICD9:035")
    other = _holder("MONDO:0000002", "y", "ICD9:035")
    [verdict] = check((ERYSIPELAS,), {"MONDO": _mondo(concept, other)})
    assert verdict.problem == (
        "xref MONDO:equivalentTo ICD9:035 is on MONDO:0000002, MONDO:0001266; "
        "the row declares MONDO:0001266"
    )


def test_a_code_row_whose_concept_holds_its_xref_under_another_qualifier_disagrees() -> None:
    concept = _holder("MONDO:0001266", "erysipelas", "ICD9:035", sources=("DOID:8154",))
    [verdict] = check((ERYSIPELAS,), {"MONDO": _mondo(concept)})
    assert verdict.problem == (
        "mondo releases/2026-10-06 has xref ICD9:035 on MONDO:0001266 without source "
        "MONDO:equivalentTo"
    )


def test_a_row_whose_concept_lacks_its_xref_disagrees() -> None:
    row = Icd9Code("428", "MONDO:0005252", "heart failure", frozenset({"MONDO:0005252"}))
    [verdict] = check((row,), {"MONDO": _mondo(_term("MONDO:0005252", "heart failure"))})
    assert verdict.problem == "mondo releases/2026-10-06 has no xref ICD9:428 on MONDO:0005252"


def test_render_writes_one_row_per_line_and_reads_back() -> None:
    rows = (
        Icd9Code("008", "MONDO:0000001", 'a "quoted" label', frozenset({"MONDO:0000001"})),
        Icd9Code(
            "428", "MONDO:0005252", "heart failure", frozenset({"MONDO:0005009", "MONDO:0005252"})
        ),
        Icd9Code("491", "MONDO:0000003", "x", frozenset({"MONDO:0000003", "A:1", "A:2"})),
    )
    module = render(rows)
    assert '    ("008", "MONDO:0000001", "a \\"quoted\\" label"),\n' in module
    assert '    "428": ("MONDO:0005009",),\n' in module
    assert '    "491": ("A:1", "A:2"),\n' in module
    namespace: dict[str, object] = {}
    exec(compile(module, "icd9_codes.py", "exec"), namespace)  # ruff: ignore[exec-builtin]
    assert namespace["ICD9_CODES"] == rows
    assert "_ALSO: Final[Mapping[str, tuple[str, ...]]] = {}\n" in render(rows[:1])


@pytest.mark.parametrize("char", ["\n", "\r", "\x00", "\x1f", "\t", "\x7f", "\x85"])
def test_render_refuses_a_control_character_naming_the_code(char: str) -> None:
    row = Icd9Code("250.41", "MONDO:0000001", f"two{char}lines", frozenset({"MONDO:0000001"}))
    with pytest.raises(ValueError, match=r"ICD-9 '250\.41': a control character"):
        render((ERYSIPELAS, row))


def test_the_committed_module_is_what_render_writes() -> None:
    committed = (Path(__file__).parents[1] / "src/bko/ontology/icd9_codes.py").read_text()
    assert committed == render(ICD9_CODES)


_MONDO_TEXT = """\
format-version: 1.2
data-version: releases/2026-10-06
ontology: mondo

[Term]
id: MONDO:0001266
name: erysipelas
xref: ICD9:035 {source="MONDO:equivalentTo", source="DOID:8154"}

[Term]
id: MONDO:0000001
name: first
xref: ICD9:558 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0000002
name: second
xref: ICD9:558 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0000009
name: obsolete y
is_obsolete: true
xref: ICD9:999 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0005249
name: pneumonia
xref: ICD9:486 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0005300
name: chronic kidney disease
xref: ICD9:585 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0005002
name: chronic obstructive pulmonary disease
xref: ICD9:496 {source="MONDO:relatedTo", source="MONDO:i2s"}

[Term]
id: MONDO:0001134
name: essential hypertension
xref: ICD9:401 {source="DOID:10825"}
"""
"""A release where 035 maps, 558's holders are not nested, 999's only holder is obsolete, 496 is
related only, 401 unreviewed only, and 486 and 585 map but are held only outside the cohort."""


def _line(encounter: int, patient: int, *diagnoses: str, **fields: str) -> str:
    diag_1, diag_2, diag_3 = (*diagnoses, "?", "?", "?")[:3]
    return diabetic_line(
        encounter_id=str(encounter),
        patient_nbr=str(patient),
        diag_1=diag_1,
        diag_2=diag_2,
        diag_3=diag_3,
        **fields,
    )


_COHORT_LINES = (
    _line(1, 1, "35", readmitted="<30"),
    _line(2, 2, "?", "558", "V45"),
    _line(3, 3, "999", "496", "401"),
    # A repeat encounter and an expired discharge leave the cohort, and their codes with them.
    _line(4, 1, "486"),
    _line(5, 5, "585", discharge_disposition_id="11"),
)


def _codes_context(mondo_text: str, zipped: bytes) -> Context:
    mondo_pin, hpo_pin = pin(mondo_text.encode(), name="mondo"), pin(b"", name="hpo")
    diabetes_pin = pin(zipped, name="uci-diabetes-130")
    opener = RoutingOpener({mondo_pin.url: mondo_text.encode(), diabetes_pin.url: zipped})
    return Context(
        open_url=opener, releases={"MONDO": mondo_pin, "HP": hpo_pin}, diabetes=diabetes_pin
    )


def test_the_codes_command_maps_the_cohorts_codes_and_logs_every_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    context = _codes_context(_MONDO_TEXT, archive(text(*_COHORT_LINES)))
    out = io.StringIO()
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code = main(["codes", "--data-dir", str(tmp_path)], context=context, out=out)
    assert code == 0
    assert out.getvalue() == render((ERYSIPELAS,))
    messages = [r.getMessage() for r in caplog.records if r.name == "bko.ontology"]
    unmapped = [m for m in messages if m.startswith("codes unmapped")]
    assert unmapped == [
        'codes unmapped reason="no live xref" codes=0',
        'codes unmapped reason="obsolete holder only" codes=1',
        'codes unmapped reason="relatedTo only" codes=1',
        'codes unmapped reason="unreviewed source only" codes=1',
        'codes unmapped reason="holders not nested" codes=1',
        'codes unmapped reason="V code" codes=1',
        'codes unmapped reason="E code" codes=0',
    ]
    assert any(m.startswith("codes rows=1 sha256=") for m in messages)


def test_a_failing_codes_run_writes_nothing(tmp_path: Path) -> None:
    changed = _codes_context(_MONDO_TEXT, archive(text(*_COHORT_LINES)))
    changed = replace(changed, releases={**changed.releases, "MONDO": pin(b"other", name="mondo")})
    out = io.StringIO()
    assert main(["codes", "--data-dir", str(tmp_path / "a")], context=changed, out=out) == 3
    assert out.getvalue() == ""
    # The release loads; the cohort, with no readmission in under 30 days, is refused after it.
    unlabelled = _codes_context(_MONDO_TEXT, archive(text(_line(1, 1, "35"))))
    assert main(["codes", "--data-dir", str(tmp_path / "b")], context=unlabelled, out=out) == 3
    assert out.getvalue() == ""
