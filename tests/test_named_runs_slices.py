"""`slice_of`: each registered holdout slice maps to the run it slices, and no other run does."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.named_runs import RUNS, NamedRun, slice_of, sliced


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

LETTERS = "abcdefghij"


@pytest.mark.parametrize("fraction", ["thin", "empty"])
@pytest.mark.parametrize("letter", list(LETTERS))
def test_each_e6_slice_maps_to_its_fractions_run(fraction, letter):
    assert slice_of(f"e6-{fraction}{letter}") == f"e6-{fraction}"


@pytest.mark.parametrize("name", ["e3-a", "e3-b"])
def test_each_e3_slice_maps_to_e3(name):
    assert slice_of(name) == "e3"


@pytest.mark.parametrize("name", ["e1", "e6-thin", "e6-empty", "e6-curated", "e3", "unregistered"])
def test_a_run_that_slices_no_other_maps_to_none(name):
    assert slice_of(name) is None


def test_a_slice_with_another_setting_than_its_whole_is_no_slice():
    whole, part = RUNS["e6-thin"], RUNS["e6-thina"]
    assert sliced(part, whole)
    other = dataclasses.replace(part.experiment, fill_to=whole.experiment.fill_to + 1)
    assert not sliced(NamedRun(other, part.settings, part.dataset), whole)
    assert not sliced(whole, part)
