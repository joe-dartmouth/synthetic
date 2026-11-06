"""`wholes_of`: each registered holdout slice maps to the runs it slices, and no other run does."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.named_runs import RUNS, NamedRun, sliced, wholes_of


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

LETTERS = "abcdefghij"


KEYS = [
    "e1",
    "g3-cell",
    "g3-coherent",
    "g3-findings",
    "g3-ontology",
    "g3-profile",
    "g3-summary",
    "g4-low",
    "g4-medium",
    "e3",
    "e3-a",
    "e3-b",
    "e4-thirty",
    "e4-forty",
    "e4-eighty",
    "e5-quarter",
    "e5-half",
    "e5-threequarters",
    "g2",
    "b1-uci",
    "b1-synthea",
    "b2-uci",
    "b2-synthea",
    "b3-uci",
    "b3-synthea",
    "g6-sonnet",
    "g6-gemma",
    "g6-nemotron",
    "g6-olmo",
    "g6-gemmathink",
    "g6-nemotronthink",
    "e6-thin",
    "e6-thina",
    "e6-thinb",
    "e6-thinc",
    "e6-thind",
    "e6-thine",
    "e6-thinf",
    "e6-thing",
    "e6-thinh",
    "e6-thini",
    "e6-thinj",
    "e6-empty",
    "e6-emptya",
    "e6-emptyb",
    "e6-emptyc",
    "e6-emptyd",
    "e6-emptye",
    "e6-emptyf",
    "e6-emptyg",
    "e6-emptyh",
    "e6-emptyi",
    "e6-emptyj",
    "e9-thin",
    "e9-thina",
    "e9-thinb",
    "e9-thinc",
    "e9-thind",
    "e9-thine",
    "e9-thinf",
    "e9-thing",
    "e9-thinh",
    "e9-thini",
    "e9-thinj",
    "e9-empty",
    "e9-emptya",
    "e9-emptyb",
    "e9-emptyc",
    "e9-emptyd",
    "e9-emptye",
    "e9-emptyf",
    "e9-emptyg",
    "e9-emptyh",
    "e9-emptyi",
    "e9-emptyj",
    "e6-curated",
    "e10-pilot-thin",
    "e10-pilot-empty",
    "g5",
]
"""Every run's name in `RUNS`' order: the names before `_slices` replaced `_e6_runs`, with E9's
and E10's pilots."""


def test_the_runs_keys_are_the_literal_list():
    assert list(RUNS) == KEYS


@pytest.mark.parametrize("entry", ["e6", "e9"])
@pytest.mark.parametrize("fraction", ["thin", "empty"])
@pytest.mark.parametrize("letter", list(LETTERS))
def test_each_slice_maps_to_its_fractions_run(entry, fraction, letter):
    assert wholes_of(f"{entry}-{fraction}{letter}") == {f"{entry}-{fraction}"}


@pytest.mark.parametrize("fraction", ["thin", "empty"])
def test_an_e9_run_is_its_e6_run_with_the_two_direct_arms_and_no_paid_arm(fraction):
    e6, e9 = RUNS[f"e6-{fraction}"], RUNS[f"e9-{fraction}"]
    assert (e9.settings, e9.dataset) == (e6.settings, e6.dataset)
    assert e9.experiment == dataclasses.replace(
        e6.experiment,
        arms=("none", "real", "bko-direct", "bko-logic"),
        replica_arms=(),
        replica_rows=None,
        candidate="bko-logic",
        comparators=("none", "bko-direct"),
    )


@pytest.mark.parametrize("name", ["e3-a", "e3-b"])
def test_each_e3_slice_maps_to_e3(name):
    assert wholes_of(name) == {"e3"}


@pytest.mark.parametrize(
    "name", ["e1", "e6-thin", "e6-empty", "e9-thin", "e9-empty", "e6-curated", "e3", "unregistered"]
)
def test_a_run_that_slices_no_other_maps_to_nothing(name):
    assert wholes_of(name) == frozenset()


def test_a_slice_with_another_setting_than_its_whole_is_no_slice():
    whole, part = RUNS["e6-thin"], RUNS["e6-thina"]
    assert sliced(part, whole)
    other = dataclasses.replace(part.experiment, fill_to=whole.experiment.fill_to + 1)
    assert not sliced(NamedRun(other, part.settings, part.dataset), whole)
    assert not sliced(whole, part)


def test_a_slice_with_other_settings_or_another_dataset_is_no_slice():
    whole, part = RUNS["e6-thin"], RUNS["e6-thina"]
    settings = dataclasses.replace(part.settings, max_tokens=part.settings.max_tokens + 1)
    assert not sliced(NamedRun(part.experiment, settings, part.dataset), whole)
    dataset = next(d for d in Dataset if d is not part.dataset)
    assert not sliced(NamedRun(part.experiment, part.settings, dataset), whole)


def test_a_slice_of_two_wholes_maps_to_both():
    twin = {**RUNS, "e6-twin": RUNS["e6-thin"]}
    assert wholes_of("e6-thina", twin) == {"e6-thin", "e6-twin"}
