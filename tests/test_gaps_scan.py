"""The first-flag scan: where each detector first flags a truth item as an ablation thins it."""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import date
from fractions import Fraction
from pathlib import Path

import pytest

import bko.gaps.scan
from bko.datasets.ablation import AblationSpec, ablate, cases, removal_order
from bko.datasets.errors import DatasetError
from bko.datasets.synthea import Population, load_population
from bko.gaps.definitions import Definition, StatementGap
from bko.gaps.detect import Statement, ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.experiment import REMOVED_FRACTIONS, run_ablation
from bko.gaps.scan import FirstFlag, log_first_flags, scan
from bko.ontology.obo import parse_obo
from tests._datasets_fixtures import synthea_tables
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

CHAIN = (
    Statement("mondo", "MONDO:9000002", "MONDO:9000001"),
    Statement("mondo", "MONDO:9000003", "MONDO:9000002"),
)
type Tables = tuple[list[dict[str, str]], list[dict[str, str]]]
W01_AT = 16
"""Where the one woman with the condition falls in the condition's seed-0 draw of 21 cases."""


def population(directory: Path, tables: Tables) -> Population:
    synthea_tables(directory, *tables)
    return load_population(directory, snapshot_date=date(2026, 1, 1))


@pytest.fixture
def fragments(tmp_path):
    mondo = parse_obo(write_obo(tmp_path, LADDER_MONDO, "mondo.obo"), name="mondo")
    return ontology_fragments({"MONDO": mondo}, {ISCHEMIA.concept})


@pytest.fixture
def people(tmp_path) -> Population:
    return population(tmp_path / "population", E2_POPULATION)


def by(flags: tuple[FirstFlag, ...], detector: str) -> dict[StatementGap, FirstFlag]:
    """A definition's flags by their pair."""
    return {
        flag.item: flag
        for flag in flags
        if flag.detector == detector and isinstance(flag.item, StatementGap)
    }


def counted(flags: tuple[FirstFlag, ...]) -> dict[tuple[str, str, int], FirstFlag]:
    """The count-based detector's flags by their cell."""
    return {
        flag.item: flag
        for flag in flags
        if flag.detector == "count" and not isinstance(flag.item, StatementGap)
    }


def test_removal_order_reproduces_every_fractions_draw(people):
    for spec in (CELL, CONDITION):
        for seed in range(4):
            selected = cases(people, spec)
            order = removal_order(selected, seed)
            for step in range(1, 101):
                fraction = step / 100
                drawn = ablate(people, dataclasses.replace(spec, fraction=fraction, seed=seed))
                prefix = order[: math.ceil(Fraction(str(fraction)) * len(selected))].sort_values()
                assert list(drawn.removed) == list(prefix)


@pytest.mark.parametrize("seed", [0, 1, 2, 5])
@pytest.mark.parametrize("fraction", REMOVED_FRACTIONS)
@pytest.mark.parametrize("whole", [CELL, CONDITION], ids=["cell", "condition"])
def test_a_monotone_detector_flags_at_a_fraction_exactly_from_its_first_flag(
    people, fragments, whole, fraction, seed
):
    spec = dataclasses.replace(whole, seed=seed)
    flags = scan("x", people, spec, fragments, [ISCHEMIA])
    partial = run_ablation(
        "x", people, dataclasses.replace(spec, fraction=fraction), fragments, [ISCHEMIA]
    )
    removed = math.ceil(Fraction(str(fraction)) * len(cases(people, spec)))
    after: dict[str, set[object]] = {
        definition.value: set(partial.after[definition])
        for definition in (Definition.SUBGROUP, Definition.POPULATION)
    }
    after["count"] = {gap.cell for gap in partial.count_after}
    assert {flag.detector for flag in flags} >= set(after)
    for flag in flags:
        if flag.detector in after:
            expected = flag.first_flag is not None and flag.first_flag <= removed
            assert (flag.item in after[flag.detector]) == expected, flag


def test_per_subgroup_flags_a_cell_only_when_its_last_case_goes(people, fragments):
    flags = scan("condition", people, CONDITION, fragments, [ISCHEMIA])
    subgroup = by(flags, Definition.SUBGROUP.value)
    assert set(subgroup) == {
        StatementGap(cell, statement)
        for cell in ((">=60", "female"), (">=60", "male"))
        for statement in CHAIN
    }
    for flag in subgroup.values():
        assert flag.cell_removed == flag.cases
        assert flag.flagged_at_all
        assert flag.reversals == 0
    woman = subgroup[StatementGap((">=60", "female"), CHAIN[0])]
    assert (woman.cases, woman.truth_from, woman.first_flag) == (1, W01_AT, W01_AT)
    men = subgroup[StatementGap((">=60", "male"), CHAIN[1])]
    assert (men.cases, men.truth_from, men.first_flag) == (20, 1, 21)


def test_population_level_flags_only_when_the_last_case_goes(people, fragments):
    condition = by(scan("condition", people, CONDITION, fragments, [ISCHEMIA]), "population-level")
    assert {flag.first_flag for flag in condition.values()} == {21}
    assert {(item.cell, flag.cell_removed) for item, flag in condition.items()} == {
        ((">=60", "female"), 1),
        ((">=60", "male"), 20),
    }
    cell = by(scan("cell", people, CELL, fragments, [ISCHEMIA]), "population-level")
    for flag in cell.values():
        assert (flag.first_flag, flag.cell_removed, flag.flagged_at_all) == (None, None, False)
        assert (flag.cases, flag.truth_from) == (20, 1)


def test_the_count_flags_from_the_first_removal_or_before_it(people, fragments):
    count = counted(scan("condition", people, CONDITION, fragments, [ISCHEMIA]))
    woman, men = count[">=60", "female", 1], count[">=60", "male", 1]
    # One case is under the threshold before anything goes, so the cell is flagged throughout.
    assert (woman.cases, woman.truth_from, woman.first_flag, woman.cell_removed) == (
        1,
        W01_AT,
        0,
        0,
    )
    assert (men.cases, men.truth_from, men.first_flag, men.cell_removed) == (20, 1, 1, 1)


def test_frequency_aware_counts_each_time_it_unlists_an_item(tmp_path, fragments):
    people = population(tmp_path / "frequency", E2_FREQUENCY_POPULATION)
    flags = scan("condition", people, CONDITION, fragments, [ISCHEMIA])
    frequency = by(flags, Definition.FREQUENCY.value)
    men = [flag for item, flag in frequency.items() if item.cell == (">=60", "male")]
    assert len(men) == 2
    full = run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA])
    for flag in men:
        # Flagged as the men's cases go while the women keep the reference share at one, then
        # unlisted once no reference patient holds the concept.
        assert flag.flagged_at_all
        assert flag.reversals == 1
        # First listed at the 31st case, which leaves 5 of the cell's 9 patients holding.
        assert (flag.first_flag, flag.cell_removed) == (31, 15)
        assert flag.item not in full.after[Definition.FREQUENCY]
    for detector in ("per-subgroup", "population-level", "count"):
        assert {flag.reversals for flag in flags if flag.detector == detector} == {0}


def test_items_follow_the_full_removals_truth_in_detector_order(people, fragments):
    flags = scan("condition", people, CONDITION, fragments, [ISCHEMIA])
    full = run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA])
    detectors = [*(definition.value for definition in Definition), "count"]
    assert [flag.detector for flag in flags] == [
        detector
        for detector in detectors
        for _ in (full.count_truth if detector == "count" else full.truth)
    ]
    assert [flag.item for flag in flags if flag.detector == "per-subgroup"] == list(full.truth)
    assert [flag.item for flag in flags if flag.detector == "count"] == list(full.count_truth)


@pytest.mark.parametrize("fraction", [0.5, 0.999])
def test_a_partial_ablation_is_refused(people, fragments, fraction):
    spec = dataclasses.replace(CONDITION, fraction=fraction)
    with pytest.raises(GapError, match="fraction 1"):
        scan("condition-50", people, spec, fragments, [ISCHEMIA])


def test_scan_refuses_a_spec_that_ablate_refuses(people, fragments):
    spec = AblationSpec(codes=CELL.codes, age_band="65-70", sex=1, edges=(60,))
    with pytest.raises(DatasetError, match="age band"):
        scan("x", people, spec, fragments, [ISCHEMIA])


def test_the_scan_silences_fusion_lines_and_restores_the_level(people, fragments, caplog):
    fusion = logging.getLogger("bko.bkb.fusion")
    fusion.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.INFO, logger="bko"):
            scan("condition", people, CONDITION, fragments, [ISCHEMIA])
            assert not [r for r in caplog.records if r.name == "bko.bkb.fusion"]
            assert fusion.level == logging.DEBUG
            with pytest.raises(GapError):
                scan("x", people, dataclasses.replace(CONDITION, fraction=0.5), fragments, [])
            run_ablation("condition", people, CONDITION, fragments, [ISCHEMIA])
        assert fusion.level == logging.DEBUG
        assert [r for r in caplog.records if r.name == "bko.bkb.fusion"]
    finally:
        fusion.setLevel(logging.NOTSET)


def test_a_failing_scan_restores_the_fusion_level(people, fragments, monkeypatch):
    fusion = logging.getLogger("bko.bkb.fusion")

    def broken(*_: object) -> None:
        assert fusion.level == logging.WARNING
        raise RuntimeError("mid-scan")

    monkeypatch.setattr(bko.gaps.scan, "_listings", broken)
    with pytest.raises(RuntimeError, match="mid-scan"):
        scan("condition", people, CONDITION, fragments, [ISCHEMIA])
    assert fusion.level == logging.NOTSET


def test_the_scan_logs_a_header_even_for_an_ablation_with_no_truth(tmp_path, people, caplog):
    mondo = parse_obo(write_obo(tmp_path, LADDER_MONDO, "mondo.obo"), name="mondo")
    unseen = ontology_fragments({"MONDO": mondo}, {UNSEEN.concept})
    with caplog.at_level(logging.INFO, logger="bko.gaps"):
        scan("condition", people, CONDITION, unseen, [UNSEEN])
    assert (
        "first-flag ablation=condition cases=21 truth per-subgroup=0 population-level=0 "
        "frequency-aware=0 count=2"
    ) in caplog.text


WOMAN = (">=60", "female")


def test_reversals_and_first_flags_are_read_from_every_listing(
    people, fragments, caplog, monkeypatch
):
    """A hand-built listing sequence: the real fixtures never relist an item, so this pins it."""
    early, late = (StatementGap(WOMAN, statement) for statement in CHAIN)
    on = {2, 5, 0}
    states = [
        {
            "per-subgroup": set(),
            "population-level": set(),
            "frequency-aware": {*({early} if nth in on else ()), *({late} if nth >= 8 else ())},
            "count": set(),
        }
        for nth in range(22)
    ]
    monkeypatch.setattr(bko.gaps.scan, "_listings", lambda *_: states)
    with caplog.at_level(logging.INFO, logger="bko.gaps"):
        flags = scan("condition", people, CONDITION, fragments, [ISCHEMIA])
        log_first_flags("condition", flags)
    frequency = by(flags, Definition.FREQUENCY.value)
    # Listed at 0, off at 1, on at 2, off at 3, on at 5, off at 6: three reversals.
    assert frequency[early].reversals == 3
    assert (frequency[early].first_flag, frequency[early].cell_removed) == (0, 0)
    assert (frequency[late].first_flag, frequency[late].reversals) == (8, 0)
    assert frequency[late].cell_removed == 0
    assert (
        "first-flag ablation=condition detector=frequency-aware cell=>=60/female cases=1 "
        "items=2 flagged=2 first=0 last=8"
    ) in caplog.text
