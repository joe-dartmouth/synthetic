"""E18: the synthetic-share sweep's runs on E6's frame, one per share, each in ten slices."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.experiment import PAID_ARMS, Experiment
from bko.evaluate.named_runs import RUNS, wholes_of
from bko.evaluate.reference import REFERENCE, reference_of
from bko.evaluate.runs_diabetes import E6_SETTINGS, E6_THIN
from bko.evaluate.runs_direct import E18, E18_ARMS, E18_SHARES


NAMES = ("two", "ten", "quarter", "half", "all")

CELL = ("6-7", "male", 1)


def test_the_shares_and_the_arms_are_the_stated_sets():
    assert list(E18_SHARES.items()) == [
        ("two", Fraction(1, 50)),
        ("ten", Fraction(1, 10)),
        ("quarter", Fraction(1, 4)),
        ("half", Fraction(1, 2)),
        ("all", Fraction(1)),
    ]
    assert E18_ARMS == (
        "real",
        "bko-direct",
        "ctgan",
        "tvae",
        "cart",
        "bayes-net",
        "tabddpm",
        "label-bootstrap",
    )
    assert tuple(E18) == NAMES
    changed = {
        "arms",
        "candidate",
        "comparators",
        "replica_arms",
        "replica_rows",
        "removal",
        "flagged_cell",
        "synthetic_share",
    }
    for name, experiment in E18.items():
        assert experiment.arms == E18_ARMS
        assert experiment.candidate == "bko-direct"
        assert experiment.comparators == (
            "ctgan",
            "tvae",
            "cart",
            "bayes-net",
            "tabddpm",
            "label-bootstrap",
        )
        assert experiment.replica_arms == ()
        assert experiment.replica_rows is None
        assert experiment.removal is None
        assert experiment.flagged_cell == CELL
        assert experiment.synthetic_share == E18_SHARES[name]
        assert experiment.training_fraction == 1
        assert experiment.decontaminate is False
        for field in dataclasses.fields(Experiment):
            if field.name not in changed:
                assert getattr(experiment, field.name) == getattr(E6_THIN, field.name), field.name


def test_no_e18_run_holds_a_paid_arm():
    names = [name for name in RUNS if name.startswith("e18-")]
    assert len(names) == 5 * 11
    for name in names:
        assert not set(RUNS[name].experiment.arms) & PAID_ARMS


@pytest.mark.parametrize("name", NAMES)
def test_each_e18_runs_slices_cover_its_holdouts_once(name):
    whole = RUNS[f"e18-{name}"]
    assert (whole.experiment, whole.settings, whole.dataset) == (
        E18[name],
        E6_SETTINGS,
        Dataset.DIABETES,
    )
    assert whole.experiment.holdout_seeds == tuple(range(100))
    held: list[int] = []
    for letter in "abcdefghij":
        assert wholes_of(f"e18-{name}{letter}") == {f"e18-{name}"}
        held += RUNS[f"e18-{name}{letter}"].experiment.holdout_seeds
    assert held == list(range(100))
    assert wholes_of(f"e18-{name}") == frozenset()


@pytest.mark.parametrize("name", NAMES)
def test_the_reference_of_a_share_run_is_bko_direct_and_e6s_cell(name):
    assert REFERENCE[f"e18-{name}"] == "bko-direct"
    for run in (f"e18-{name}", f"e18-{name}c"):
        assert reference_of(run, RUNS[run].experiment, RUNS) == ("bko-direct", CELL)
