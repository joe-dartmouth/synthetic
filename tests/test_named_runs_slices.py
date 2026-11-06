"""`wholes_of`: each registered holdout slice maps to the runs it slices, and no other run does."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.named_runs import RUNS, NamedRun, sliced, wholes_of


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

LETTERS = "abcdefghij"


@pytest.mark.parametrize("fraction", ["thin", "empty"])
@pytest.mark.parametrize("letter", list(LETTERS))
def test_each_e6_slice_maps_to_its_fractions_run(fraction, letter):
    assert wholes_of(f"e6-{fraction}{letter}") == {f"e6-{fraction}"}


@pytest.mark.parametrize("name", ["e3-a", "e3-b"])
def test_each_e3_slice_maps_to_e3(name):
    assert wholes_of(name) == {"e3"}


@pytest.mark.parametrize("name", ["e1", "e6-thin", "e6-empty", "e6-curated", "e3", "unregistered"])
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
