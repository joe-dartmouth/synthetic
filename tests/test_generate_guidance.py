"""Guidance from gaps: each cell's findings, their basis, and the rows that ground them."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import Statement, grounded, ontology_fragments
from bko.gaps.observe import Cell, CellRecords, Cells, Observed, dataset_fragment, record_cells
from bko.generate.guidance import (
    Basis,
    Finding,
    Guidance,
    Profile,
    Relationship,
    Share,
    Spread,
    guide,
    intervals,
)
from bko.ontology.obo import parse_obo
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology
from tests._gaps_fixtures import ANGINA, LADDER_HPO, STENOSIS, parse_ladders
from tests._ontology_fixtures import write_obo
from tests._validate_fixtures import BASE


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

A = ANGINA.concept
B = STENOSIS.concept
H21 = Statement("hpo", "HP:9000002", "HP:9000001")
H32 = Statement("hpo", "HP:9000003", "HP:9000002")
M21 = Statement("mondo", "MONDO:9000002", "MONDO:9000001")
M32 = Statement("mondo", "MONDO:9000003", "MONDO:9000002")
M42 = Statement("mondo", "MONDO:9000004", "MONDO:9000002")
ROWS = (ANGINA, STENOSIS)

YOUNG_SICK = ("<55", "female", 1)
OLD_WELL = (">=55", "male", 0)
YOUNG_WELL = ("<55", "male", 0)

EXERTION = FieldMapping(
    when=(Codes("exang", frozenset({1})),),
    concept=A,
    label="Angina pectoris",
    source="uci-heart-disease",
    limits="test",
)
CHEST = FieldMapping(
    when=(Codes("cp", frozenset({1, 2, 3})),),
    concept="HP:9000002",
    label="Chest pain",
    source="uci-heart-disease",
    limits="test",
)

SICK_ANGINA = FieldMapping(
    when=(Codes(LABEL, frozenset({1})), Codes("cp", frozenset({1}))),
    concept=B,
    label="coronary stenosis",
    source="uci-heart-disease",
    limits="test",
)
"""A row that reads the label beside another field, so it is not met by every record."""

H21_LABELLED = Relationship(H21, "Chest pain", "Phenotypic abnormality")
H32_LABELLED = Relationship(H32, "Angina pectoris", "Chest pain")


@pytest.fixture
def ladders(tmp_path) -> dict[str, Ontology]:
    return parse_ladders(tmp_path)


def record(**changes: int | float) -> dict[str, int | float]:
    """`BASE`, a 63-year-old man without the condition and with typical angina, changed."""
    return {**BASE, "exang": 0, **changes}


def frame(records: Iterable[Mapping[str, int | float]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


TWO_YOUNG_SICK = [record(age=50, sex=0, target=1, cp=4)] * 2
"""Two young women with the condition and no angina: `YOUNG_SICK` lacks the HPO chain."""

OLD_WELL_ANGINA = [record()]
"""One older man without the condition, with typical angina: `OLD_WELL` has no gap."""

BLANKS = {"older": record(cp=4), "young": record(age=50, cp=4)}
"""A man without the condition who shows no concept, one per age band."""


def with_both_bands(
    records: Sequence[Mapping[str, int | float]],
) -> list[Mapping[str, int | float]]:
    """`records` plus a blank record (no concept, label 0) in each age band they lack.

    Each cell's age interval intersects its band with the split's ages, so the split needs
    an age in every band.
    """
    older = any(one["age"] >= 55 for one in records)
    young = any(one["age"] < 55 for one in records)
    return [*records, *([] if older else [BLANKS["older"]]), *([] if young else [BLANKS["young"]])]


def observed(
    records: Sequence[Mapping[str, int | float]], rows: Sequence[FieldMapping] = ROWS
) -> tuple[pd.DataFrame, Cells]:
    """The training frame of `records` and its cells."""
    train = frame(records)
    return train, record_cells(train, rows, name="uci")


def guided(
    train: pd.DataFrame,
    cells: Cells,
    found: Iterable[StatementGap],
    ontologies: Mapping[str, Ontology],
    *,
    definition: Definition = Definition.SUBGROUP,
    rows: Sequence[FieldMapping] = ROWS,
) -> dict[Cell, Guidance]:
    fragments = ontology_fragments(ontologies, {row.concept for row in rows})
    return guide(
        train,
        cells,
        found,
        definition=definition,
        fragments=fragments,
        ontologies=ontologies,
        rows=rows,
        frame=UCI_FRAME,
    )


def subgroup(cells: Cells, ontologies: Mapping[str, Ontology]) -> tuple[StatementGap, ...]:
    return gaps(Definition.SUBGROUP, ontology_fragments(ontologies, {A, B}), cells)


def statements(guidance: Guidance) -> list[Statement]:
    return sorted(r.statement for f in guidance.findings for r in f.relationships)


def test_subgroup_gaps_become_absent_findings_named_by_release_label_and_field_state(ladders):
    train, cells = observed([*TWO_YOUNG_SICK, *OLD_WELL_ANGINA])
    guidance = guided(train, cells, subgroup(cells, ladders), ladders)[YOUNG_SICK]
    assert guidance.cell == YOUNG_SICK
    assert (guidance.sex, guidance.label, guidance.records) == (0, 1, 2)
    assert guidance.interval == intervals(YOUNG_SICK, train, frame=UCI_FRAME)
    assert guidance.basis is Basis.ABSENT
    assert guidance.findings == (Finding((ANGINA,), (H21_LABELLED, H32_LABELLED)),)
    assert (guidance.profile, guidance.summary) == (None, None)


def meeting(row: FieldMapping) -> dict[str, int | float]:
    """A young woman with the condition, no chest pain and no exertional angina, made to meet `row`.

    A coded clause takes its highest code, so the chest-pain row is met by cp 3, not by angina.
    """
    met = record(age=50, sex=0, target=1, cp=4)
    for clause in row.when:
        met[clause.field] = max(clause.codes) if isinstance(clause, Codes) else clause.value
    return met


def test_a_record_meeting_a_finding_grounds_every_statement_rendered_for_it(ladders):
    rows = (ANGINA, EXERTION, CHEST, STENOSIS)
    train, cells = observed([*TWO_YOUNG_SICK, *OLD_WELL_ANGINA], rows=rows)
    findings = guided(train, cells, subgroup(cells, ladders), ladders, rows=rows)[
        YOUNG_SICK
    ].findings
    assert [len(finding.states) for finding in findings] == [2, 3]
    for finding in findings:
        rendered = {relationship.statement for relationship in finding.relationships}
        for row in finding.states:
            met = meeting(row)
            assert row.applies(met)
            after = subgroup(observed([*TWO_YOUNG_SICK, met, *OLD_WELL_ANGINA], rows)[1], ladders)
            left = [gap for gap in after if gap.cell == YOUNG_SICK and gap.statement in rendered]
            assert not left, row


FREQUENCY_RECORDS = (
    [record(age=63, sex=0, target=0, cp=1)]
    + [record(age=63, sex=0, target=0, cp=4)] * 19
    + [record(age=63, sex=1, target=0, cp=1)] * 20
    + [record(age=63, sex=0, target=1, cp=1)] * 5
    + [record(age=63, sex=1, target=1, cp=4)] * 5
)
"""Older women without the condition rarely show angina, as `test_gaps_definitions` sets out."""

NO_ANGINA = [record(cp=4), record(age=50, sex=0, target=1, cp=4)]
"""No record shows angina, so the HPO chain is a population-level gap in every cell."""


@pytest.mark.parametrize(
    ("definition", "records", "basis"),
    [
        (Definition.SUBGROUP, FREQUENCY_RECORDS, Basis.ABSENT),
        (Definition.POPULATION, NO_ANGINA, Basis.ABSENT),
        (Definition.FREQUENCY, FREQUENCY_RECORDS, Basis.RARE),
    ],
)
def test_each_definition_gets_its_basis(ladders, definition, records, basis):
    train, cells = observed(with_both_bands(records))
    found = gaps(definition, ontology_fragments(ladders, {A, B}), cells)
    assert found
    guidance = guided(train, cells, found, ladders, definition=definition)
    with_gaps = {gap.cell for gap in found}
    for cell in with_gaps:
        assert guidance[cell].basis is basis, cell
        assert guidance[cell].findings, cell
    without = [cell for cell in cells.each if cell not in with_gaps]
    shown = [cell for cell in without if cells.each[cell].observed.held]
    assert {guidance[cell].basis for cell in shown} <= {Basis.SHOWN}
    assert shown or definition is Definition.POPULATION


def test_statements_reached_by_the_same_rows_group_into_one_finding(ladders):
    train, cells = observed(with_both_bands(TWO_YOUNG_SICK), rows=(ANGINA, EXERTION, STENOSIS))
    found = [StatementGap(YOUNG_SICK, H21), StatementGap(YOUNG_SICK, H32)]
    rows = (ANGINA, EXERTION, STENOSIS)
    assert guided(train, cells, found, ladders, rows=rows)[YOUNG_SICK].findings == (
        Finding((ANGINA, EXERTION), (H21_LABELLED, H32_LABELLED)),
    )
    split = (ANGINA, EXERTION, CHEST)
    assert guided(train, cells, found, ladders, rows=split)[YOUNG_SICK].findings == (
        Finding((ANGINA, EXERTION), (H32_LABELLED,)),
        Finding((ANGINA, EXERTION, CHEST), (H21_LABELLED,)),
    )


def test_a_row_needing_another_label_is_not_offered_in_this_cell(ladders, caplog):
    rows = (ANGINA, SICK_ANGINA)
    train, cells = observed([*TWO_YOUNG_SICK, *OLD_WELL_ANGINA], rows=rows)
    found = [StatementGap(YOUNG_SICK, M42), StatementGap(OLD_WELL, M42)]
    with caplog.at_level(logging.WARNING, logger="bko"):
        guidance = guided(train, cells, found, ladders, definition=Definition.POPULATION, rows=rows)
    assert guidance[YOUNG_SICK].findings == (
        Finding((SICK_ANGINA,), (Relationship(M42, "coronary stenosis", "heart disorder"),)),
    )
    assert guidance[OLD_WELL].findings == ()
    assert f"unreached cell=>=55/male/0 statement={M42}" in caplog.messages


def test_a_statement_no_compatible_row_reaches_is_dropped_with_a_warning(ladders, caplog):
    train, cells = observed(with_both_bands(TWO_YOUNG_SICK))
    found = [StatementGap(YOUNG_SICK, M32), StatementGap(YOUNG_SICK, H32)]
    with caplog.at_level(logging.WARNING, logger="bko"):
        guidance = guided(train, cells, found, ladders)[YOUNG_SICK]
    assert guidance.basis is Basis.ABSENT
    assert statements(guidance) == [H32]
    assert [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("unreached")
    ] == [("WARNING", f"unreached cell=<55/female/1 statement={M32}")]


def test_a_cell_with_no_gap_is_guided_by_what_its_records_ground(ladders):
    train, cells = observed([*TWO_YOUNG_SICK, *OLD_WELL_ANGINA])
    found = subgroup(cells, ladders)
    assert OLD_WELL not in {gap.cell for gap in found}
    guidance = guided(train, cells, found, ladders)[OLD_WELL]
    fragments = ontology_fragments(ladders, {A, B})
    expected = grounded(fragments, dataset_fragment(cells.each[OLD_WELL].observed))
    assert expected
    assert guidance.basis is Basis.SHOWN
    assert statements(guidance) == list(expected)
    assert guidance.findings == (Finding((ANGINA,), (H21_LABELLED, H32_LABELLED)),)


def test_a_statement_a_label_only_row_reaches_is_met_by_every_record_and_skipped(ladders, caplog):
    train, cells = observed(with_both_bands([record(age=50, sex=0, target=1, cp=1)] * 2))
    assert YOUNG_SICK not in {gap.cell for gap in subgroup(cells, ladders)}
    with caplog.at_level(logging.DEBUG, logger="bko"):
        guidance = guided(train, cells, subgroup(cells, ladders), ladders)[YOUNG_SICK]
    assert guidance.basis is Basis.SHOWN
    assert guidance.findings == (Finding((ANGINA,), (H21_LABELLED, H32_LABELLED)),)
    assert not [r for r in caplog.records if r.getMessage().startswith("unreached")]
    assert f"pinned cell=<55/female/1 statement={M42}" in caplog.messages


def test_a_cell_with_no_gap_and_no_shown_concept_gets_basis_none(ladders):
    train, cells = observed([record(age=50, cp=4), *OLD_WELL_ANGINA])
    guidance = guided(train, cells, (), ladders)
    for cell in (YOUNG_WELL, ("<55", "female", 0)):
        assert guidance[cell].basis is Basis.NONE
        assert guidance[cell].findings == ()
    assert guidance[YOUNG_WELL].records == 1


def test_a_gap_cell_outside_the_grid_or_a_cell_without_a_label_is_refused(ladders):
    train, cells = observed(TWO_YOUNG_SICK)
    with pytest.raises(ValueError, match=r"\('x',\) is not in the grid"):
        guided(train, cells, [StatementGap(("x",), H21)], ladders)
    unlabelled = Cells(
        whole=Observed(name="s", records=1, holders={A: 1}),
        each={("60-69", "male"): CellRecords(Observed("s", 1, {A: 1}), None, frozenset())},
    )
    with pytest.raises(ValueError, match="60-69"):
        guided(frame(TWO_YOUNG_SICK), unlabelled, (), ladders)


def test_a_row_whose_prefix_has_no_release_is_refused(ladders):
    train, cells = observed(TWO_YOUNG_SICK)
    stray = FieldMapping(
        when=(Codes(LABEL, frozenset({1})),),
        concept="DOID:1",
        label="stray",
        source="uci-heart-disease",
        limits="test",
    )
    with pytest.raises(ValueError, match="DOID"):
        guide(
            train,
            cells,
            (),
            definition=Definition.SUBGROUP,
            fragments=ontology_fragments(ladders, {A, B}),
            ontologies=ladders,
            rows=(*ROWS, stray),
            frame=UCI_FRAME,
        )


def test_a_curie_with_no_name_is_labelled_by_its_curie(tmp_path, ladders):
    nameless = LADDER_HPO.replace("name: Chest pain\n", "")
    ontologies = {**ladders, "HP": parse_obo(write_obo(tmp_path, nameless, "bare.obo"), name="hpo")}
    train, cells = observed(with_both_bands(TWO_YOUNG_SICK))
    found = [StatementGap(YOUNG_SICK, H21), StatementGap(YOUNG_SICK, H32)]
    assert guided(train, cells, found, ontologies)[YOUNG_SICK].findings == (
        Finding(
            (ANGINA,),
            (
                Relationship(H21, "HP:9000002", "Phenotypic abnormality"),
                Relationship(H32, "Angina pectoris", "HP:9000002"),
            ),
        ),
    )
    unreleased = Statement("hpo", A, "XO:1")
    (finding,) = guided(train, cells, [StatementGap(YOUNG_SICK, unreleased)], ladders)[
        YOUNG_SICK
    ].findings
    assert finding.relationships == (Relationship(unreleased, "Angina pectoris", "XO:1"),)


def test_gap_order_does_not_change_the_guidance(ladders):
    train, cells = observed(with_both_bands(FREQUENCY_RECORDS))
    found = subgroup(cells, ladders)
    assert len(found) > 1

    def read(guidance: Mapping[Cell, Guidance]) -> list[tuple[Cell, Basis, tuple[Finding, ...]]]:
        return [(cell, one.basis, one.findings) for cell, one in guidance.items()]

    assert read(guided(train, cells, reversed(found), ladders)) == read(
        guided(train, cells, found, ladders)
    )
    assert list(guided(train, cells, reversed(found), ladders)) == list(cells.each)


WELL_ISCHEMIA = FieldMapping(
    when=(Codes(LABEL, frozenset({0})), Codes("fbs", frozenset({1}))),
    concept="MONDO:9000003",
    label="myocardial ischemia",
    source="uci-heart-disease",
    limits="test",
)
"""A row only records without the condition meet, so no label-1 cell is offered it."""

PROFILE_ROWS = (ANGINA, EXERTION, CHEST, STENOSIS, WELL_ISCHEMIA)

REFERENCE = [
    record(age=60, target=1, cp=1, fbs=0, chol=200),
    record(age=70, target=1, cp=4, exang=1, fbs=0, chol=240),
    record(age=80, target=1, cp=4, fbs=1, chol=280),
    record(age=40, target=1, cp=4, fbs=0, chol=300),
]
"""The men with the condition: `YOUNG_SICK`'s reference group, three older and one younger.

Angina (cp 1 or exang 1) is shown by 2 of 4, chest pain (cp 1 to 3) by 1 of 4."""

OUTSIDE_THE_REFERENCE = [
    *[record(age=50, sex=0, target=1, cp=1, fbs=0, chol=100)] * 2,
    record(),
    record(age=40, cp=4, fbs=1, chol=500),
]
"""`YOUNG_SICK`'s own records, and two men without the condition: none is a reference record."""


def test_profile_is_each_compatible_concepts_share_among_records_outside_the_cell_with_its_label(
    ladders,
):
    train, cells = observed([*REFERENCE, *OUTSIDE_THE_REFERENCE], rows=PROFILE_ROWS)
    guidance = guided(train, cells, (), ladders, rows=PROFILE_ROWS)[YOUNG_SICK]
    assert guidance.profile == Profile(
        4,
        (
            Share(A, "Angina pectoris", (ANGINA, EXERTION), 0.5),
            Share("HP:9000002", "Chest pain", (CHEST,), 0.25),
        ),
    )


def test_the_summary_is_each_numeric_fields_quartiles_over_the_same_records(ladders):
    train, cells = observed([*REFERENCE, *OUTSIDE_THE_REFERENCE], rows=PROFILE_ROWS)
    guidance = guided(train, cells, (), ladders, rows=PROFILE_ROWS)[YOUNG_SICK]
    assert guidance.summary == {
        "age": Spread(55.0, 65.0, 72.5),
        "trestbps": Spread(145.0, 145.0, 145.0),
        "chol": Spread(230.0, 260.0, 285.0),
        "thalach": Spread(150.0, 150.0, 150.0),
        "oldpeak": Spread(2.3, 2.3, 2.3),
    }


def test_a_cell_with_no_reference_record_gets_no_profile_or_summary_and_a_warning(ladders, caplog):
    train, cells = observed([*TWO_YOUNG_SICK, *OLD_WELL_ANGINA])
    with caplog.at_level(logging.WARNING, logger="bko"):
        guidance = guided(train, cells, (), ladders)
    assert (guidance[YOUNG_SICK].profile, guidance[YOUNG_SICK].summary) == (None, None)
    assert "noreference cell=<55/female/1" in caplog.messages
    assert "noreference cell=>=55/male/0" in caplog.messages
    young_well = guidance[YOUNG_WELL]
    assert young_well.profile == Profile(1, (Share(A, "Angina pectoris", (ANGINA,), 1.0),))
    assert young_well.summary is not None
    assert young_well.summary["age"] == Spread(63.0, 63.0, 63.0)


def test_a_training_frame_other_than_the_one_observed_is_refused(ladders):
    train, cells = observed([*REFERENCE, *OUTSIDE_THE_REFERENCE], rows=PROFILE_ROWS)
    other = frame([*REFERENCE, REFERENCE[0], *OUTSIDE_THE_REFERENCE])
    with pytest.raises(ValueError, match=r"5 reference rows .* 4 reference records"):
        guided(other, cells, (), ladders, rows=PROFILE_ROWS)
    assert guided(train, cells, (), ladders, rows=PROFILE_ROWS)[YOUNG_SICK].profile is not None
