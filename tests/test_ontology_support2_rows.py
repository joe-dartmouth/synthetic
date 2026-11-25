"""SUPPORT2's field rows against its frame's codes and the citation registry; no pronto."""

from __future__ import annotations

from bko.datasets.support2 import CITED, TERTILES, frame_records, parse_support2
from bko.datasets.support2_frame import support2_frame
from bko.ontology.mapping import CHECKED_ROWS, CONCEPT_ROWS
from bko.ontology.schema import Codes
from bko.ontology.support2_rows import SUPPORT2_FIELD_ROWS
from tests._registry import REGISTRY, read, registry_entries
from tests._support2_fixtures import mixed_rows, text


def _frame():
    return support2_frame(frame_records(parse_support2(text(*mixed_rows()))))


def test_the_rows_map_these_states_to_these_concepts() -> None:
    found = []
    for row in SUPPORT2_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        found.append((clause.field, sorted(clause.codes), row.concept, row.label))
    assert found == [
        ("dzgroup", [1], "MONDO:0005002", "chronic obstructive pulmonary disease"),
        ("dzgroup", [2], "MONDO:0005009", "congestive heart failure"),
        ("dzgroup", [3], "MONDO:0005155", "cirrhosis of liver"),
        ("dzgroup", [5], "MONDO:0021063", "malignant colon neoplasm"),
        ("dzgroup", [6], "MONDO:0008903", "lung cancer"),
        ("dzgroup", [4], "HP:0001259", "Coma"),
        ("diabetes", [1], "MONDO:0005015", "diabetes mellitus"),
        ("dementia", [1], "MONDO:0001627", "dementia"),
        ("ca", [1, 2], "MONDO:0004992", "cancer"),
        ("hrt", [2], "HP:0001649", "Tachycardia"),
        ("resp", [2], "HP:0002789", "Tachypnea"),
        ("temp", [3], "HP:0001945", "Fever"),
        ("temp", [1], "HP:0002045", "Hypothermia"),
        ("wblc", [3], "HP:0001974", "Increased total leukocyte count"),
        ("wblc", [1], "HP:0001882", "Decreased total leukocyte count"),
        ("meanbp", [1], "HP:0002615", "Hypotension"),
        ("crea", [2], "HP:0003259", "Elevated circulating creatinine concentration"),
        ("bili", [2], "HP:0002904", "Hyperbilirubinemia"),
        ("pafi", [1], "HP:0012418", "Hypoxemia"),
    ]


def test_each_row_names_a_coded_field_and_its_codes_not_state_0() -> None:
    frame = _frame()
    for row in SUPPORT2_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        assert clause.field in frame.categorical, clause.field
        assert clause.codes <= set(frame.codebook[clause.field]), row
        if clause.field in CITED:
            assert 0 not in clause.codes, row


def test_no_row_reads_scoma_or_a_tertile_field() -> None:
    read_fields = {clause.field for row in SUPPORT2_FIELD_ROWS for clause in row.when}
    assert read_fields.isdisjoint({"scoma", *TERTILES})


def test_every_cutoff_row_cites_a_registered_source() -> None:
    keys = {key for _, key, _ in registry_entries(read(REGISTRY))}
    cutoff_rows = [row for row in SUPPORT2_FIELD_ROWS if any(c.field in CITED for c in row.when)]
    assert len(cutoff_rows) == 10
    for row in cutoff_rows:
        assert row.source == "singer-2016", row
        assert "read in singer-2016's reproduction" in row.limits, row
        assert ("bone-1992" in row.limits) != ("vincent-1996" in row.limits), row
    for key in ("singer-2016", "bone-1992", "vincent-1996", "vanderbilt-2026"):
        assert key in keys, key
    assert {row.source for row in SUPPORT2_FIELD_ROWS} <= keys


def test_the_rows_join_the_checked_table() -> None:
    assert CONCEPT_ROWS[-len(SUPPORT2_FIELD_ROWS) :] == SUPPORT2_FIELD_ROWS
    assert set(SUPPORT2_FIELD_ROWS) <= set(CHECKED_ROWS)
