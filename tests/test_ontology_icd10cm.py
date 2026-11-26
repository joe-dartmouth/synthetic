"""`bko.ontology.icd10`: NHAMCS's ICD-10-CM token rule, its generated rows and their command."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from bko.datasets.icd10_tokens import BLANK, UNKNOWN, token_number, token_text
from bko.ontology.__main__ import Context, main
from bko.ontology.check import check
from bko.ontology.codes import render
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.icd10 import REASONS, Icd10Code, generate_icd10, icd10cm_id, render_icd10
from bko.ontology.icd10cm_codes import ICD10CM_CODES
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from tests._datasets_fixtures import pin
from tests._nhamcs_fixtures import BLANK_CHRONIC, chronic, line, nhamcs_archive
from tests._ontology_fixtures import RoutingOpener
from tests.test_ontology_check import _mondo
from tests.test_ontology_codes import _holder


ICD9_MODULE_SHA256 = "4a2f84dec3a5b114d3f5e549ae702cd9506e98e004fe7935e5e1e6d11c0382d7"
"""The sha256 of the committed `bko.ontology.icd9_codes`."""

HEART_FAILURE = Icd10Code("I50.9", "MONDO:0005252", "heart failure", frozenset({"MONDO:0005252"}))


def test_a_four_character_token_reads_as_a_dotted_code() -> None:
    assert icd10cm_id("I509") == "ICD10CM:I50.9"
    assert icd10cm_id("F32A") == "ICD10CM:F32.A"
    assert HEART_FAILURE.token == "I509"


def test_a_padded_token_reads_as_its_category() -> None:
    assert icd10cm_id("A00-") == "ICD10CM:A00"
    assert icd10cm_id("M1A-") == "ICD10CM:M1A"
    assert Icd10Code("A00", "MONDO:0000001", "cholera", frozenset()).token == "A00-"


@pytest.mark.parametrize("token", [BLANK, "ZZZ0", "ZZZ2", "ZZZ3", "ZZZ4", "ZZZ5"])
def test_blank_and_non_icd_tokens_map_nothing(token: str) -> None:
    assert icd10cm_id(token) is None
    holder = _holder("MONDO:0000001", "x", "ICD10CM:I50.9")
    rows, unmapped = generate_icd10(_mondo(holder), [token])
    assert rows == []
    assert {reason: n for reason, n in unmapped.items() if n} == (
        {} if token == BLANK else {"not ICD-10-CM": 1}
    )


@pytest.mark.parametrize("token", ["-9", "ZZZ1", "i509", "I50.9", "1509", "IA09", "I5-9", ""])
def test_a_token_the_file_cannot_hold_is_refused(token: str) -> None:
    with pytest.raises(ValueError, match="is not an NHAMCS diagnosis token"):
        icd10cm_id(token)
    with pytest.raises(ValueError, match="is not an NHAMCS diagnosis token"):
        token_number(token)


def test_only_exact_cross_references_map() -> None:
    exact = _holder("MONDO:0000001", "exact", "ICD10CM:I50.9")
    related = _holder("MONDO:0000002", "related", "ICD10CM:I50.9", sources=("MONDO:relatedTo",))
    weak = _holder("MONDO:0000003", "weak", "ICD10CM:J45", sources=("MONDO:relatedTo",))
    unreviewed = _holder("MONDO:0000004", "unreviewed", "ICD10CM:K21", sources=("DOID:8534",))
    rows, unmapped = generate_icd10(
        _mondo(exact, related, weak, unreviewed), ["I509", "J45-", "K21-", "R509"]
    )
    assert rows == [Icd10Code("I50.9", "MONDO:0000001", "exact", frozenset({"MONDO:0000001"}))]
    assert tuple(unmapped) == REASONS
    assert dict(unmapped) == {
        "no live xref": 1,
        "obsolete holder only": 0,
        "relatedTo only": 1,
        "unreviewed source only": 1,
        "holders not nested": 0,
        "not ICD-10-CM": 0,
    }


def test_a_token_maps_only_its_truncated_id_never_a_longer_code() -> None:
    longer = _holder("MONDO:0000001", "longer", "ICD10CM:I50.91")
    category = _holder("MONDO:0000002", "category", "ICD10CM:I50")
    rows, unmapped = generate_icd10(_mondo(longer, category), ["I509", "I50-"])
    assert [(row.code, row.concept) for row in rows] == [("I50", "MONDO:0000002")]
    assert unmapped["no live xref"] == 1


def test_nested_holders_map_the_top_one_and_others_stay_unmapped() -> None:
    top = _holder("MONDO:0000001", "top", "ICD10CM:E11.9")
    below = _holder("MONDO:0000002", "below", "ICD10CM:E11.9", parents=("MONDO:0000001",))
    first = _holder("MONDO:0000003", "first", "ICD10CM:N39.0")
    second = _holder("MONDO:0000004", "second", "ICD10CM:N39.0")
    old = replace(_holder("MONDO:0000005", "old", "ICD10CM:R10.9"), obsolete=True)
    rows, unmapped = generate_icd10(
        _mondo(top, below, first, second, old), ["E119", "N390", "R109"]
    )
    assert rows == [
        Icd10Code("E11.9", "MONDO:0000001", "top", frozenset({"MONDO:0000001", "MONDO:0000002"}))
    ]
    assert (unmapped["holders not nested"], unmapped["obsolete holder only"]) == (1, 1)


def test_a_nameless_concept_is_refused() -> None:
    nameless = replace(_holder("MONDO:0000001", "x", "ICD10CM:I50.9"), name=None)
    with pytest.raises(ValueError, match="MONDO:0000001 has no name"):
        generate_icd10(_mondo(nameless), ["I509"])


def test_rows_follow_the_token_order() -> None:
    holders = [
        _holder(f"MONDO:000000{n}", f"t{n}", f"ICD10CM:{code}")
        for n, code in enumerate(("A00", "A00.0", "B20"), 1)
    ]
    rows, _ = generate_icd10(_mondo(*holders), ["B20-", "A000", "A00-", "A00-"])
    assert [row.token for row in rows] == ["A00-", "A000", "B20-"]


def test_the_token_number_is_reversible_and_distinct() -> None:
    assert token_number(BLANK) == UNKNOWN == 0
    assert token_number("A00-") == 11 * 37**3 + 1 * 37**2 + 1 * 37 + 0
    assert token_number("A000") == token_number("A00-") + 1
    assert token_number("ZZZ5") == 36 * 37**3 + 36 * 37**2 + 36 * 37 + 6
    for token in (BLANK, "A00-", "A000", "I509", "F32A", "M1A0", "ZZZ0", "ZZZ5", "Z992"):
        assert token_text(token_number(token)) == token


@pytest.mark.parametrize("number", [-1, 1, 37**4, token_number("A00-") - 1])
def test_a_number_no_token_encodes_is_refused(number: int) -> None:
    with pytest.raises(ValueError, match="is not an encoded NHAMCS diagnosis token"):
        token_text(number)


def test_the_icd9_rows_keep_their_bytes() -> None:
    committed = (Path(__file__).parents[1] / "src/bko/ontology/icd9_codes.py").read_bytes()
    assert hashlib.sha256(committed).hexdigest() == ICD9_MODULE_SHA256
    assert render(ICD9_CODES).encode() == committed


def test_render_writes_one_row_per_line_and_reads_back() -> None:
    rows = (
        Icd10Code("A00", "MONDO:0000001", 'a "quoted" label', frozenset({"MONDO:0000001"})),
        Icd10Code("I50.9", "MONDO:0005252", "x", frozenset({"MONDO:0005252", "MONDO:0000009"})),
    )
    module = render_icd10(rows)
    assert '    ("A00", "MONDO:0000001", "a \\"quoted\\" label"),\n' in module
    assert '    "I50.9": ("MONDO:0000009",),\n' in module
    namespace: dict[str, object] = {}
    exec(compile(module, "icd10cm_codes.py", "exec"), namespace)  # ruff: ignore[exec-builtin]
    assert namespace["ICD10CM_CODES"] == rows


def test_render_refuses_a_control_character_naming_the_code() -> None:
    row = Icd10Code("I50.9", "MONDO:0000001", "two\nlines", frozenset({"MONDO:0000001"}))
    with pytest.raises(ValueError, match=r"^ICD-10-CM 'I50\.9': a control character"):
        render_icd10((row,))


def test_the_committed_module_is_what_render_writes() -> None:
    committed = (Path(__file__).parents[1] / "src/bko/ontology/icd10cm_codes.py").read_text()
    assert committed == render_icd10(ICD10CM_CODES)
    assert ICD10CM_CODES
    assert len({row.token for row in ICD10CM_CODES}) == len(ICD10CM_CODES)
    assert all(row.concept.startswith("MONDO:") for row in ICD10CM_CODES)


def test_the_generated_rows_join_the_checked_table() -> None:
    assert set(ICD10CM_CODES) <= set(CONCEPT_ROWS)
    assert set(ICD10CM_CODES) <= set(CHECKED_ROWS)


def test_a_code_row_is_checked_against_its_exact_holders() -> None:
    concept = _holder("MONDO:0005252", "heart failure", "ICD10CM:I50.9")
    [verdict] = check((HEART_FAILURE,), {"MONDO": _mondo(concept)})
    assert (verdict.table, verdict.subject, verdict.problem) == (
        "icd10cm-code",
        "ICD10CM:I50.9",
        None,
    )
    other = _holder("MONDO:0000002", "y", "ICD10CM:I50.9")
    [verdict] = check((HEART_FAILURE,), {"MONDO": _mondo(concept, other)})
    assert verdict.problem == (
        "xref MONDO:equivalentTo ICD10CM:I50.9 is on MONDO:0000002, MONDO:0005252; "
        "the row declares MONDO:0005252"
    )


_MONDO_TEXT = """\
format-version: 1.2
data-version: releases/2026-10-06
ontology: mondo

[Term]
id: MONDO:0005252
name: heart failure
xref: ICD10CM:I50.9 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0004979
name: asthma
xref: ICD10CM:J45 {source="MONDO:relatedTo"}

[Term]
id: MONDO:0005249
name: pneumonia
xref: ICD10CM:J18.9 {source="MONDO:equivalentTo"}
"""
"""A release where I50.9 maps, J45 is related only, and J18.9 maps but no visit holds it."""


def _context(*lines: str) -> Context:
    zipped = nhamcs_archive(*lines)
    mondo_pin, hpo_pin = pin(_MONDO_TEXT.encode(), name="mondo"), pin(b"", name="hpo")
    nhamcs_pin = pin(zipped, name="nhamcs-ed-2022")
    opener = RoutingOpener({mondo_pin.url: _MONDO_TEXT.encode(), nhamcs_pin.url: zipped})
    return Context(open_url=opener, releases={"MONDO": mondo_pin, "HP": hpo_pin}, nhamcs=nhamcs_pin)


def test_the_codes_command_maps_the_records_tokens_and_logs_every_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    context = _context(
        line(1, ADMITHOS="1", DIAG1="I509", DIAG2="J45-", **chronic("CHF")),
        line(2, DIAG1="ZZZ0", DIAG3="R509", **BLANK_CHRONIC),
    )
    out = io.StringIO()
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code = main(
            ["codes", "--system", "icd10cm", "--data-dir", str(tmp_path)], context=context, out=out
        )
    assert code == 0
    assert out.getvalue() == render_icd10((HEART_FAILURE,))
    messages = [r.getMessage() for r in caplog.records if r.name == "bko.ontology"]
    assert [m for m in messages if m.startswith("codes unmapped")] == [
        'codes unmapped system=icd10cm reason="no live xref" codes=1',
        'codes unmapped system=icd10cm reason="obsolete holder only" codes=0',
        'codes unmapped system=icd10cm reason="relatedTo only" codes=1',
        'codes unmapped system=icd10cm reason="unreviewed source only" codes=0',
        'codes unmapped system=icd10cm reason="holders not nested" codes=0',
        'codes unmapped system=icd10cm reason="not ICD-10-CM" codes=1',
    ]
    assert any(m.startswith("codes system=icd10cm rows=1 tokens=4 sha256=") for m in messages)


def test_a_failing_icd10cm_codes_run_writes_nothing(tmp_path: Path) -> None:
    context = _context(line(1, ADMITHOS="0"))
    out = io.StringIO()
    argv = ["codes", "--system", "icd10cm", "--data-dir", str(tmp_path)]
    assert main(argv, context=context, out=out) == 3
    assert out.getvalue() == ""
