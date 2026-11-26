"""E20: `bko-micro` on E6's frame, `E9_THIN` and `E9_EMPTY` with it the candidate, in ten slices."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.direct_arms import CAPPED_ARM, MICRO_ARM
from bko.evaluate.experiment import DIRECT_ARM, PAID_ARMS, REAL_ARM
from bko.evaluate.named_runs import RUNS, wholes_of
from bko.evaluate.reference import REFERENCE, reference_of
from bko.evaluate.runs_diabetes import E6_SETTINGS, E9_EMPTY, E9_THIN
from bko.evaluate.runs_direct import E20_EMPTY, E20_THIN


PAIRS = (("thin", E20_THIN, E9_THIN), ("empty", E20_EMPTY, E9_EMPTY))


@pytest.mark.parametrize(("name", "micro", "e9"), PAIRS)
def test_e20_is_e9_with_the_micro_arm_against_capped_direct_and_none(name, micro, e9):
    assert micro.arms == ("none", REAL_ARM, DIRECT_ARM, CAPPED_ARM, MICRO_ARM)
    assert micro.candidate == MICRO_ARM
    assert micro.comparators == (DIRECT_ARM, CAPPED_ARM, "none")
    assert micro.replica_arms == ()
    changed = {"arms", "candidate", "comparators"}
    for field in dataclasses.fields(micro):
        if field.name not in changed:
            assert getattr(micro, field.name) == getattr(e9, field.name), field.name
    assert not set(micro.arms) & PAID_ARMS
    assert RUNS[f"e20-{name}"].experiment == micro


def test_the_e20_names_are_the_wholes_and_their_slices():
    names = [name for name in RUNS if name.startswith("e20-")]
    wholes = ("e20-thin", "e20-empty")
    assert names == [whole + suffix for whole in wholes for suffix in ("", *"abcdefghij")]


@pytest.mark.parametrize(("name", "micro", "e9"), PAIRS)
def test_each_e20_runs_slices_cover_its_holdouts_once(name, micro, e9):
    whole = RUNS[f"e20-{name}"]
    assert (whole.settings, whole.dataset) == (E6_SETTINGS, Dataset.DIABETES)
    assert whole.experiment.holdout_seeds == e9.holdout_seeds
    held: list[int] = []
    for letter in "abcdefghij":
        assert wholes_of(f"e20-{name}{letter}") == {f"e20-{name}"}
        held += RUNS[f"e20-{name}{letter}"].experiment.holdout_seeds
    assert held == list(micro.holdout_seeds)
    assert wholes_of(f"e20-{name}") == frozenset()


@pytest.mark.parametrize(("name", "micro", "e9"), PAIRS)
def test_the_reference_of_an_e20_run_is_bko_direct(name, micro, e9):
    assert REFERENCE[f"e20-{name}"] == DIRECT_ARM
    assert micro.removal is not None
    for run in (f"e20-{name}", f"e20-{name}c"):
        assert reference_of(run, RUNS[run].experiment, RUNS) == (DIRECT_ARM, micro.removal.cell)
