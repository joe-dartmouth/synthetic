"""`bko.ontology.codes`: the generated ICD-9 code rows, their rule, their text and their command."""

from __future__ import annotations

import io
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from bko.ontology.__main__ import Context, main
from bko.ontology.check import check
from bko.ontology.codes import REASONS, generate, render
from bko.ontology.icd9 import ICD9_MAPPINGS, Icd9Code
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Term
from tests._datasets_fixtures import pin
from tests._diabetes_fixtures import archive, diabetic_line, text
from tests._ontology_fixtures import RoutingOpener, write_obo
from tests.test_ontology_check import _mondo, _term


def _holder(curie: str, name: str, *xrefs: str, parents: tuple[str, ...] = ()) -> Term:
    return replace(_term(curie, name), xrefs=tuple(sorted(xrefs)), parents=parents)


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


def test_codes_without_a_holder_and_v_and_e_codes_are_counted_by_reason() -> None:
    v_holder = _holder("MONDO:0000001", "v holder", "ICD9:V45")
    generated = generate(_mondo(v_holder), ["?", "V45", "V45.81", "E888.1", "123.4"])
    assert generated.rows == ()
    assert tuple(generated.unmapped) == REASONS
    assert dict(generated.unmapped) == {
        "no live xref": 1,
        "obsolete holder only": 0,
        "holders not nested": 0,
        "V code": 2,
        "E code": 1,
    }


def test_a_concept_without_a_name_is_refused() -> None:
    nameless = replace(_holder("MONDO:0000001", "x", "ICD9:123"), name=None)
    with pytest.raises(ValueError, match="MONDO:0000001 has no name"):
        generate(_mondo(nameless), ["123"])


def test_every_code_maps_to_one_concept() -> None:
    assert ICD9_CODES
    assert len({row.code for row in ICD9_CODES}) == len(ICD9_CODES)
    for row in ICD9_CODES:
        assert row.concept in row.holders, row
        assert row.concept.startswith("MONDO:"), row


def test_the_curated_rows_win_in_the_committed_table() -> None:
    committed = {row.xref: row.concept for row in ICD9_CODES}
    for curated in ICD9_MAPPINGS:
        assert committed.get(curated.xref, curated.concept) == curated.concept, curated


def test_the_generated_rows_join_the_checked_table() -> None:
    assert set(ICD9_CODES) <= set(CONCEPT_ROWS)
    assert set(ICD9_CODES) <= set(CHECKED_ROWS)


def _stanzas(rows: tuple[Icd9Code, ...]) -> str:
    """MONDO stanzas carrying each row's xref on each of its holders, named by the row's label."""
    xrefs: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for row in rows:
        for curie in row.holders:
            xrefs.setdefault(curie, set()).add(row.xref)
            names.setdefault(curie, row.label if curie == row.concept else f"holder {curie}")
        names[row.concept] = row.label
    body = "".join(
        f"\n[Term]\nid: {curie}\nname: {names[curie]}\n"
        + "".join(f"xref: {xref}\n" for xref in sorted(xrefs[curie]))
        for curie in sorted(xrefs)
    )
    return f"format-version: 1.2\ndata-version: releases/2026-10-06\nontology: mondo\n{body}"


def test_every_committed_row_names_a_live_term_carrying_its_xref(tmp_path: Path) -> None:
    mondo = parse_obo(write_obo(tmp_path, _stanzas(ICD9_CODES), "mondo.obo"), name="mondo")
    verdicts = check(ICD9_CODES, {"MONDO": mondo})
    assert [v.problem for v in verdicts] == [None] * len(ICD9_CODES)
    assert {v.table for v in verdicts} == {"icd9-code"}
    assert verdicts[0].subject == ICD9_CODES[0].xref


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


def test_the_committed_module_is_what_render_writes() -> None:
    committed = (Path(__file__).parents[1] / "src/bko/ontology/icd9_codes.py").read_text()
    assert committed == render(ICD9_CODES)


def test_the_codes_command_writes_the_module_and_logs_its_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mondo_text = _stanzas(ICD9_CODES[:2])
    lines = (
        diabetic_line(encounter_id="1", patient_nbr="1", diag_1=ICD9_CODES[0].code.lstrip("0")),
        diabetic_line(encounter_id="2", patient_nbr="2", diag_2=ICD9_CODES[1].code, diag_3="V45"),
    )
    zipped = archive(text(*lines))
    mondo_pin, hpo_pin = pin(mondo_text.encode(), name="mondo"), pin(b"", name="hpo")
    diabetes_pin = pin(zipped, name="uci-diabetes-130")
    opener = RoutingOpener({mondo_pin.url: mondo_text.encode(), diabetes_pin.url: zipped})
    context = Context(
        open_url=opener, releases={"MONDO": mondo_pin, "HP": hpo_pin}, diabetes=diabetes_pin
    )
    out = io.StringIO()
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code = main(["codes", "--data-dir", str(tmp_path)], context=context, out=out)
    assert code == 0
    assert out.getvalue() == render(ICD9_CODES[:2])
    messages = [r.getMessage() for r in caplog.records if r.name == "bko.ontology"]
    assert 'codes unmapped reason="V code" codes=1' in messages
    assert 'codes unmapped reason="no live xref" codes=3' in messages
    assert any(m.startswith("codes rows=2 sha256=") for m in messages)
