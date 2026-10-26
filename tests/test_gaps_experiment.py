"""E2: what each gap definition and the count gain from a removal, read against its truth."""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path

import pytest

from bko.datasets.ablation import DEFAULT_ABLATION, AblationSpec
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, Population, load_population
from bko.datasets.synthea_run import DEFAULT_RUN
from bko.gaps.definitions import Definition, StatementGap
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.experiment import (
    E2_ABLATIONS,
    E2_RUN,
    REMOVED_FRACTIONS,
    Detection,
    Recovery,
    detection,
    recovery,
    run_ablation,
)
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Ontology
from tests._datasets_fixtures import condition, patient, synthea_tables
from tests._gaps_fixtures import (
    CELL,
    CONDITION,
    E2_FREQUENCY_POPULATION,
    E2_POPULATION,
    ISCHEMIA,
    LADDER_MONDO,
    UNSEEN,
)
from tests._ontology_fixtures import write_obo


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ISCHEMIA_CHAIN = (
    Statement("mondo", "MONDO:9000002", "MONDO:9000001"),
    Statement("mondo", "MONDO:9000003", "MONDO:9000002"),
)
EVERY_CELL = (("<60", "female"), ("<60", "male"), (">=60", "female"), (">=60", "male"))
type Tables = tuple[list[dict[str, str]], list[dict[str, str]]]


def pairs(*cells: tuple[str, str]) -> tuple[StatementGap, ...]:
    """The ischemia chain in each of `cells`, in the order given."""
    return tuple(StatementGap(cell, statement) for cell in cells for statement in ISCHEMIA_CHAIN)


def population(directory: Path, tables: Tables) -> Population:
    synthea_tables(directory, *tables)
    return load_population(directory, snapshot_date=date(2026, 1, 1))


@pytest.fixture
def mondo(tmp_path) -> Ontology:
    return parse_obo(write_obo(tmp_path, LADDER_MONDO, "mondo.obo"), name="mondo")


def test_recovery_counts_the_truth_gaps_absent_before_and_present_after():
    reading = recovery(before={"a", "b"}, after={"a", "c", "d"}, truth={"b", "c", "e"})
    assert reading == Recovery(
        truth=3, eligible=2, recovered=1, outside=2, new_outside=1, vanished=1
    )


def test_recovery_with_nothing_eligible_reads_zero_of_zero():
    assert recovery(before={"a"}, after={"a"}, truth={"a"}) == Recovery(1, 0, 0, 0, 0, 0)


def test_detection_reads_sensitivity_and_specificity_against_the_truth():
    reading = detection(
        before={"a"}, after={"a", "b", "x"}, truth={"a", "b", "c", "d"}, universe=set("abcdexyz")
    )
    # eligible b, c, d (a was flagged before); only b recovered; outside e, x, y, z, with x new
    assert reading == Detection(sensitivity=1 / 3, specificity=0.75)


def test_detection_with_nothing_eligible_or_outside_reads_none():
    assert detection(before={"a"}, after={"a"}, truth={"a"}, universe={"a"}) == Detection(
        None, None
    )


def test_e2_removes_one_cell_then_the_whole_condition_from_the_default_population():
    assert list(E2_ABLATIONS) == [
        "cell",
        "condition",
        *(
            f"{name}-{round(fraction * 100)}"
            for name in ("cell", "condition")
            for fraction in REMOVED_FRACTIONS
        ),
    ]
    assert REMOVED_FRACTIONS == (0.25, 0.5, 0.75)
    for name in ("cell", "condition"):
        for fraction in REMOVED_FRACTIONS:
            partial = E2_ABLATIONS[f"{name}-{round(fraction * 100)}"]
            assert partial == dataclasses.replace(E2_ABLATIONS[name], fraction=fraction)
    assert E2_ABLATIONS["cell"] == DEFAULT_ABLATION
    condition = E2_ABLATIONS["condition"]
    assert (condition.codes, condition.age_band, condition.sex, condition.edges) == (
        DEFAULT_ABLATION.codes,
        None,
        None,
        DEFAULT_ABLATION.edges,
    )
    assert E2_RUN == DEFAULT_RUN


def test_a_cell_removal_is_found_by_per_subgroup_grounding_and_the_count(tmp_path, mondo):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("cell", people, CELL, fragments, [ISCHEMIA])
    assert list(run.before_cells.each) == list(EVERY_CELL)
    assert run.truth == pairs((">=60", "male"))
    assert run.before[Definition.SUBGROUP] == pairs(("<60", "female"), ("<60", "male"))
    assert run.after[Definition.SUBGROUP] == pairs(
        ("<60", "female"), ("<60", "male"), (">=60", "male")
    )
    assert run.reading(Definition.SUBGROUP) == Recovery(
        truth=2, eligible=2, recovered=2, outside=4, new_outside=0, vanished=0
    )
    assert (run.before[Definition.POPULATION], run.after[Definition.POPULATION]) == ((), ())
    assert run.reading(Definition.POPULATION) == Recovery(2, 2, 0, 0, 0, 0)
    assert run.before[Definition.FREQUENCY] == pairs(("<60", "female"))
    assert run.after[Definition.FREQUENCY] == pairs(("<60", "female"))
    assert run.reading(Definition.FREQUENCY) == Recovery(2, 2, 0, 2, 0, 0)
    assert run.count_truth == ((">=60", "male", 1),)
    assert len(run.count_before) == 6
    assert run.count == Recovery(
        truth=1, eligible=1, recovered=1, outside=6, new_outside=0, vanished=0
    )


def test_a_whole_condition_removal_puts_population_level_gaps_new_outside_the_truth(
    tmp_path, mondo
):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA])
    assert run.truth == pairs((">=60", "female"), (">=60", "male"))
    assert run.after[Definition.SUBGROUP] == pairs(*EVERY_CELL)
    assert run.after[Definition.POPULATION] == pairs(*EVERY_CELL)
    assert run.reading(Definition.SUBGROUP) == Recovery(4, 4, 4, 4, 0, 0)
    assert run.reading(Definition.POPULATION) == Recovery(4, 4, 4, 4, 4, 0)
    assert run.after[Definition.FREQUENCY] == ()
    assert run.reading(Definition.FREQUENCY) == Recovery(4, 4, 0, 0, 0, 2)
    assert run.count_truth == ((">=60", "female", 1), (">=60", "male", 1))
    assert run.count == Recovery(2, 1, 1, 5, 0, 0)


def test_a_cell_keeping_patients_who_show_nothing_is_found_by_the_frequency_aware_score(
    tmp_path, mondo
):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_FREQUENCY_POPULATION)
    run = run_ablation("cell", people, CELL, fragments, [ISCHEMIA])
    assert run.before[Definition.FREQUENCY] == ()
    assert run.after[Definition.FREQUENCY] == pairs((">=60", "male"))
    assert run.reading(Definition.FREQUENCY) == Recovery(2, 2, 2, 0, 0, 0)
    assert run.reading(Definition.POPULATION) == Recovery(2, 2, 0, 0, 0, 0)


@pytest.mark.parametrize("spec", [CELL, CONDITION], ids=["cell", "condition"])
def test_per_subgroup_and_the_count_put_nothing_new_outside_the_truth_and_lose_nothing(
    tmp_path, mondo, spec
):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept, UNSEEN.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("x", people, spec, fragments, [ISCHEMIA, UNSEEN])
    for reading in (run.reading(Definition.SUBGROUP), run.count):
        assert (reading.new_outside, reading.vanished) == (0, 0)
        assert reading.recovered == reading.eligible


def test_a_gap_before_the_removal_stays_out_of_the_truth(tmp_path, mondo):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept, UNSEEN.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA, UNSEEN])
    unseen = Statement("mondo", "MONDO:9000004", "MONDO:9000002")
    assert unseen not in {gap.statement for gap in run.truth}
    for definition in (Definition.SUBGROUP, Definition.POPULATION):
        for listed in (run.before[definition], run.after[definition]):
            assert {gap.cell for gap in listed if gap.statement == unseen} == set(EVERY_CELL)
    assert run.reading(Definition.SUBGROUP) == Recovery(4, 4, 4, 8, 0, 0)


def test_listed_pairs_follow_the_grid_order_of_the_ablations_bands(tmp_path, mondo):
    spec = AblationSpec(
        codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band=">=60", sex=1, edges=(40, 60)
    )
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("cell", people, spec, fragments, [ISCHEMIA])
    bands = ("<40", "40-59", ">=60")
    assert list(run.before_cells.each) == [(b, s) for b in bands for s in ("female", "male")]
    cells = list(dict.fromkeys(gap.cell for gap in run.listed(Definition.SUBGROUP)))
    assert cells == [
        ("<40", "female"),
        ("<40", "male"),
        ("40-59", "female"),
        ("40-59", "male"),
        (">=60", "male"),
    ]


def test_a_removal_that_leaves_no_patient_is_refused(tmp_path, mondo):
    tables: Tables = (
        [patient("m1", "1961-07-15", "M")],
        [condition("m1", ISCHEMIC_HEART_DISEASE, "2019-06-01")],
    )
    people = population(tmp_path / "population", tables)
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    with pytest.raises(GapError, match="the condition ablation removes every patient"):
        run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA])


def test_a_gap_off_the_ablations_grid_is_refused(tmp_path, mondo):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("cell", people, CELL, fragments, [ISCHEMIA])
    stray = StatementGap(("<40", "male"), ISCHEMIA_CHAIN[0])
    broken = dataclasses.replace(run, after={**run.after, Definition.SUBGROUP: (stray,)})
    with pytest.raises(GapError, match="the cell ablation lists gaps off its grid: <40/male"):
        broken.listed(Definition.SUBGROUP)


def test_a_partial_cell_removal_is_seen_by_the_count_and_not_by_grounding(tmp_path, mondo):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    half = dataclasses.replace(CELL, fraction=0.5)
    run = run_ablation("cell-50", people, half, fragments, [ISCHEMIA])
    assert len(run.ablation.removed) == 10
    # Ten men of the cell still ground every statement, so no definition flags it anew.
    assert run.detection(Definition.SUBGROUP) == Detection(sensitivity=0.0, specificity=1.0)
    assert run.detection(Definition.POPULATION).sensitivity == pytest.approx(0.0)
    # The cell's count fell from the threshold to ten, which the count flags.
    assert run.count_detection == Detection(sensitivity=1.0, specificity=1.0)


def test_a_full_removal_reads_full_sensitivity_for_grounding(tmp_path, mondo):
    fragments = ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})
    people = population(tmp_path / "population", E2_POPULATION)
    run = run_ablation("cell", people, CELL, fragments, [ISCHEMIA])
    assert run.detection(Definition.SUBGROUP) == Detection(sensitivity=1.0, specificity=1.0)
