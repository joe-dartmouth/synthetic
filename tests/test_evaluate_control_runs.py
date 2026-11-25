"""The control runs on E6's and E3's frames: their arms, holdouts, gap requests and references.

No test reaches the API: no control run holds a paid arm, and the context's send refuses.
"""

from __future__ import annotations

import pytest

from bko.datasets.split import split
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, load
from bko.evaluate.experiment import PAID_ARMS, gap_request
from bko.evaluate.named_runs import E3, RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.reference import reference_of
from bko.evaluate.run import method_arms
from bko.evaluate.runs_diabetes import E6_EMPTY, E6_THIN
from bko.evaluate.runs_direct import CONTROL_ARMS, E12_EMPTY, E12_THIN, E13
from bko.generate.messages import AnySend
from tests._evaluate_fixtures import FAST_METHODS
from tests._ontology_fixtures import RoutingOpener
from tests._synthea_frame_fixtures import (
    FRAME_RUN,
    JAR_PIN,
    RELEASE,
    RELEASE_PIN,
    write_population,
)


LETTERS = "abcdefghij"

WHOLES = ("e12-thin", "e12-empty", "e13")
"""The control runs that slice no other run."""

NAMES = (
    *WHOLES,
    *(f"e12-{fraction}{letter}" for fraction in ("thin", "empty") for letter in LETTERS),
)
"""Every control run's name, the slices included."""


def _unsendable() -> AnySend:
    msg = "no control run builds the live send"
    raise AssertionError(msg)


def _context() -> Context:
    return Context(
        open_url=RoutingOpener({RELEASE_PIN.url: RELEASE.encode()}),
        runs={"e13": RUNS["e13"]},
        arms=method_arms(FAST_METHODS),
        releases={"MONDO": RELEASE_PIN},
        send=_unsendable,
        synthea=FRAME_RUN,
        synthea_jar=JAR_PIN,
    )


def test_the_control_arms_are_the_stated_set():
    assert CONTROL_ARMS == (
        "none",
        "class-weight",
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "real",
        "bayes-net",
        "bko-direct",
        "bko-nomap",
        "label-bootstrap",
        "prior-shift",
    )


COMPARATORS = (
    "none",
    "class-weight",
    "smote",
    "adasyn",
    "ctgan",
    "tvae",
    "bayes-net",
    "bko-nomap",
    "label-bootstrap",
    "prior-shift",
)
"""Every control arm but the candidate and `real`, written out."""


@pytest.mark.parametrize(
    ("e12", "e6", "name", "e6_name"),
    [
        (E12_THIN, E6_THIN, "e12-thin", "e6-thin"),
        (E12_EMPTY, E6_EMPTY, "e12-empty", "e6-empty"),
    ],
)
def test_each_e12_run_is_e6_with_the_control_arms(e12, e6, name, e6_name):
    assert (e12.removal, e12.fill_to, e12.cutoff) == (e6.removal, e6.fill_to, e6.cutoff)
    assert e12.holdout_seeds == e6.holdout_seeds
    assert e12.arms == CONTROL_ARMS
    assert e12.candidate == "bko-direct"
    assert e12.comparators == COMPARATORS
    assert e12.replica_arms == ()
    assert e12.replica_rows is None
    assert RUNS[name].experiment == e12
    assert (RUNS[name].settings, RUNS[name].dataset) == (
        RUNS[e6_name].settings,
        Dataset.DIABETES,
    )


@pytest.mark.parametrize("name", NAMES)
def test_no_control_run_holds_a_paid_arm(name):
    assert not PAID_ARMS & set(RUNS[name].experiment.arms)


@pytest.mark.parametrize("name", ["e12-thin", "e12-empty"])
def test_the_e12_slices_cover_each_holdout_once(name):
    slices = [RUNS[f"{name}{letter}"].experiment for letter in LETTERS]
    seeds = [seed for part in slices for seed in part.holdout_seeds]
    assert len(seeds) == len(set(seeds))
    assert tuple(seeds) == RUNS[name].experiment.holdout_seeds


def test_e13_is_e3_with_the_control_arms_and_bko_logic():
    assert E13.removal == E3.removal
    assert E13.holdout_seeds == E3.holdout_seeds
    assert E13.decontaminate is False
    assert E13.arms == (*CONTROL_ARMS, "bko-logic")
    assert E13.candidate == "bko-direct"
    assert E13.comparators == (*COMPARATORS, "bko-logic")
    assert E13.replica_arms == ()
    assert E13.fill_to == 20
    assert E13.spend_ceiling == pytest.approx(1.0)
    assert (RUNS["e13"].experiment, RUNS["e13"].dataset) == (E13, Dataset.SYNTHEA)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e13s_gap_request_asks_the_removed_cell(tmp_path, seed):
    """The removed cell alone is asked for, 15 rows.

    Its 65 members split 52 to training (13 held out, one fifth rounded half up); 0.9 of them
    removed, rounded up, is 47, leaving 5, so a fill to 20 asks 15.
    """
    data_dir = tmp_path / "data"
    write_population(data_dir)
    data = load(Dataset.SYNTHEA, data_dir, _context())
    parts = split(data.records, seed=seed, edges=data.frame.edges)
    train = prepared(parts, E13, edges=data.frame.edges).ablated.train
    request = gap_request(train, E13, seed=seed, frame=data.frame)
    assert dict(request.quota) == {(">=65", "female", 1): 15}


@pytest.mark.parametrize("name", NAMES)
def test_the_reference_of_each_control_run_is_real(name):
    assert reference_of(name, RUNS[name].experiment, RUNS)[0] == "real"
