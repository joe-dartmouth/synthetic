"""P3: Juwara et al.'s removal fractions on E6's cell, each fill sized as their code sizes it."""

from __future__ import annotations

import dataclasses

import pytest

from bko.datasets.cells import cell_counts
from bko.evaluate.data import Dataset
from bko.evaluate.experiment import PAID_ARMS, Experiment, Removal, gap_request
from bko.evaluate.named_runs import RUNS, wholes_of
from bko.evaluate.prepare import prepared
from bko.evaluate.runs_diabetes import E6_SETTINGS, E6_THIN
from bko.evaluate.runs_published import P3, P3_ARMS, P3_FRACTIONS, juwara_total
from tests._frame_fixtures import TOY, TOY_PARTS


NAMES = ("fifteen", "thirty", "fifty", "eighty", "ninetyfive")

CELL = ("6-7", "male", 1)

FULL = ("<45", "male", 1)
"""A toy cell holding 20 training rows at seed 0."""


def test_the_fractions_are_juwaras():
    assert list(P3_FRACTIONS.items()) == [
        ("fifteen", 0.15),
        ("thirty", 0.30),
        ("fifty", 0.50),
        ("eighty", 0.80),
        ("ninetyfive", 0.95),
    ]


def test_the_arms_are_the_stated_set():
    assert P3_ARMS == (
        "none",
        "real",
        "smote",
        "ctgan",
        "cart",
        "bko-direct",
        "bko-nomap",
        "label-bootstrap",
        "prior-shift",
    )
    fill_to = {"fifteen": 1268, "thirty": 1268, "fifty": 1268, "eighty": 1265, "ninetyfive": 1260}
    assert tuple(P3) == NAMES
    changed = {
        "arms",
        "candidate",
        "comparators",
        "replica_arms",
        "replica_rows",
        "removal",
        "flagged_cell",
        "fill_to",
    }
    for name, experiment in P3.items():
        assert experiment.arms == P3_ARMS
        assert experiment.candidate == "bko-direct"
        assert experiment.comparators == (
            "none",
            "smote",
            "ctgan",
            "cart",
            "bko-nomap",
            "label-bootstrap",
            "prior-shift",
        )
        assert experiment.replica_arms == ()
        assert experiment.replica_rows is None
        assert experiment.removal == Removal(CELL, P3_FRACTIONS[name])
        assert experiment.flagged_cell == CELL
        assert experiment.fill_to == fill_to[name]
        for field in dataclasses.fields(Experiment):
            if field.name not in changed:
                assert getattr(experiment, field.name) == getattr(E6_THIN, field.name), field.name


@pytest.mark.parametrize(
    ("fraction", "remaining", "quota"),
    [(0.15, 17, 3), (0.30, 14, 6), (0.50, 10, 10), (0.80, 4, 16), (0.95, 1, 19)],
)
def test_the_fill_size_is_juwaras_total_less_the_rows_that_remain(fraction, remaining, quota):
    assert juwara_total(20, fraction) == 20
    experiment = Experiment(
        holdout_seeds=(0, 1),
        arms=("none", "smote"),
        candidate="smote",
        comparators=("none",),
        replica_arms=(),
        removal=Removal(FULL, fraction),
        flagged_cell=FULL,
        fill_to=juwara_total(20, fraction),
    )
    train = prepared(TOY_PARTS, experiment, edges=TOY.edges).ablated.train
    counts = cell_counts(train, edges=TOY.edges)
    kept = counts.set_index(["age_band", "sex", "target"])["count"]
    assert kept[FULL] == remaining
    request = gap_request(train, experiment, seed=0, frame=TOY)
    assert request.quota == {FULL: quota}
    assert quota == juwara_total(20, fraction) - remaining


def test_juwaras_total_floors_the_restored_count():
    assert [juwara_total(1269, fraction) for fraction in P3_FRACTIONS.values()] == [
        1268,
        1268,
        1268,
        1265,
        1260,
    ]


def test_juwaras_total_floors_where_ceiling_and_rounding_would_not():
    # 0.15 of 7 rows removes ceil(1.05) = 2, so 5 remain and 5 / 0.85 is 5.88: floored to 5.
    assert juwara_total(7, 0.15) == 5


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_juwaras_total_refuses_a_fraction_it_cannot_restore(fraction):
    with pytest.raises(ValueError, match=r"must lie strictly between 0 and 1"):
        juwara_total(20, fraction)


def test_no_p3_run_holds_a_paid_arm():
    names = [name for name in RUNS if name.startswith("p3-")]
    assert names
    for name in names:
        assert not set(RUNS[name].experiment.arms) & PAID_ARMS


def test_every_p3_run_is_a_whole_or_one_of_the_slices_that_cover_its_holdouts_once():
    held: dict[str, list[int]] = {f"p3-{name}": [] for name in NAMES}
    for name in (name for name in RUNS if name.startswith("p3-")):
        wholes = wholes_of(name)
        if not wholes:
            assert name in held
            continue
        (whole,) = wholes
        held[whole] += RUNS[name].experiment.holdout_seeds
    for whole, seeds in held.items():
        assert sorted(seeds) == list(RUNS[whole].experiment.holdout_seeds), whole


@pytest.mark.parametrize("name", NAMES)
def test_each_p3_runs_slices_cover_its_holdouts_once(name):
    whole = RUNS[f"p3-{name}"]
    assert (whole.experiment, whole.settings, whole.dataset) == (
        P3[name],
        E6_SETTINGS,
        Dataset.DIABETES,
    )
    assert whole.experiment.holdout_seeds == tuple(range(100))
    held: list[int] = []
    for letter in "abcdefghij":
        assert wholes_of(f"p3-{name}{letter}") == {f"p3-{name}"}
        held += RUNS[f"p3-{name}{letter}"].experiment.holdout_seeds
    assert held == list(range(100))
    assert wholes_of(f"p3-{name}") == frozenset()
