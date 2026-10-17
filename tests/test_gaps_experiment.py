"""E2: what each detector gains from a removal, read against the gaps the removal accounts for."""

from __future__ import annotations

from datetime import date

import pytest

from bko.datasets.ablation import DEFAULT_ABLATION
from bko.datasets.synthea import load_population
from bko.datasets.synthea_run import DEFAULT_RUN
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.experiment import E2_ABLATIONS, E2_RUN, Recovery, recovery, run_ablation
from bko.ontology.obo import parse_obo
from tests._datasets_fixtures import synthea_tables
from tests._gaps_fixtures import CELL, CONDITION, E2_POPULATION, ISCHEMIA, LADDER_MONDO
from tests._ontology_fixtures import write_obo


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ISCHEMIA_CHAIN = (
    Statement("mondo", "MONDO:9000002", "MONDO:9000001"),
    Statement("mondo", "MONDO:9000003", "MONDO:9000002"),
)


@pytest.fixture
def e2(tmp_path):
    synthea_tables(tmp_path / "population", *E2_POPULATION)
    population = load_population(tmp_path / "population", snapshot_date=date(2026, 1, 1))
    mondo = parse_obo(write_obo(tmp_path, LADDER_MONDO, "mondo.obo"), name="mondo")
    return population, ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})


def test_recovery_counts_the_truth_gaps_absent_before_and_present_after():
    reading = recovery(before={"a", "b"}, after={"a", "b", "c", "d"}, truth={"b", "c", "e"})
    assert reading == Recovery(truth=3, eligible=2, recovered=1, spurious=1)


def test_recovery_with_nothing_eligible_reads_zero_of_zero():
    assert recovery(before={"a"}, after={"a"}, truth={"a"}) == Recovery(1, 0, 0, 0)


def test_e2_removes_one_cell_then_the_whole_condition_from_the_default_population():
    assert list(E2_ABLATIONS) == ["cell", "condition"]
    assert E2_ABLATIONS["cell"] == DEFAULT_ABLATION
    condition = E2_ABLATIONS["condition"]
    assert (condition.codes, condition.age_band, condition.sex, condition.edges) == (
        DEFAULT_ABLATION.codes,
        None,
        None,
        DEFAULT_ABLATION.edges,
    )
    assert E2_RUN == DEFAULT_RUN


def test_a_cell_removal_is_invisible_to_the_bko_detector_and_found_by_the_count(e2):
    population, ontology = e2
    run = run_ablation("cell", population, CELL, ontology, [ISCHEMIA])
    assert (run.bko_before, run.bko_after, run.bko_truth) == ((), (), ISCHEMIA_CHAIN)
    assert run.bko == Recovery(truth=2, eligible=2, recovered=0, spurious=0)
    assert run.count_truth == ((">=60", "male", 1),)
    assert len(run.count_before) == 6
    assert run.count == Recovery(truth=1, eligible=1, recovered=1, spurious=0)


def test_a_whole_condition_removal_is_found_by_both(e2):
    population, ontology = e2
    run = run_ablation("condition", population, CONDITION, ontology, [ISCHEMIA])
    assert (run.bko_before, run.bko_after, run.bko_truth) == ((), ISCHEMIA_CHAIN, ISCHEMIA_CHAIN)
    assert run.bko == Recovery(truth=2, eligible=2, recovered=2, spurious=0)
    assert run.count_truth == ((">=60", "female", 1), (">=60", "male", 1))
    assert run.count == Recovery(truth=2, eligible=1, recovered=1, spurious=0)
