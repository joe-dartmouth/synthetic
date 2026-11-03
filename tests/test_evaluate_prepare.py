"""`bko.evaluate.prepare`: each holdout's training rows sized by the training fraction."""

from __future__ import annotations

import dataclasses
import io
import logging
from fractions import Fraction

import pytest

import bko.evaluate.likelihood as likelihood_module
from bko.datasets.split import Split, split, subsampled
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.classifier import Model
from bko.evaluate.experiment import E1, Removal, flagged, gap_request
from bko.evaluate.holdout_candidates import read
from bko.evaluate.prepare import fingerprint, held_out_ratio, prepared
from bko.evaluate.rehearse import safe_ranges
from bko.evaluate.run import score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL, Spy
from tests.test_evaluate_likelihood import DATA, candidate, cells, encode


HALF = dataclasses.replace(
    SMALL,
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    replica_arms=("smote",),
    training_fraction=Fraction(1, 2),
)
EDGES = UCI_FRAME.edges
PARTS = split(GRADED, seed=0, edges=EDGES)
SIZED = subsampled(PARTS.train, Fraction(1, 2), seed=0, edges=EDGES)


def test_every_stage_reads_the_subsample(caplog):
    seen: list[tuple[str, int]] = []

    def spying(name):
        def runner(train, request):
            seen.append((name, len(train)))
            return FAST_ARMS[name](train, request)

        return runner

    spy = Spy()
    arms = {name: spying(name) for name in HALF.arms}
    with caplog.at_level(logging.INFO):
        score_holdout(PARTS, HALF, arms, models={Model.LR: spy}, frame=UCI_FRAME)
    assert len(SIZED) < len(PARTS.train)
    assert {size for _, size in seen} == {len(SIZED)}
    assert len(seen) == len(HALF.arms) + len(HALF.replica_arms)
    # Every classifier fit reads the subsample's rows, plus the rows an arm adds.
    assert all(len(model.seen[0]) >= len(SIZED) for model in spy.models)
    assert min(len(model.seen[0]) for model in spy.models) == len(SIZED)
    assert f"split seed=0 train={len(SIZED)} test={len(PARTS.test)} " in "\n".join(caplog.messages)
    assert (
        f"subsample holdout=0 fraction=1/2 train={len(PARTS.train)} kept={len(SIZED)} "
        f"ratio={held_out_ratio(Split(SIZED, PARTS.test, 0), edges=EDGES):.6f}" in caplog.messages
    )


def test_held_out_ratio_reads_the_subsample():
    scores = score_holdout(PARTS, HALF, FAST_ARMS, frame=UCI_FRAME)
    (ratio,) = [
        s.value for s in scores if s.arm == HELD_OUT_ARM and s.measure is Measure.HELD_OUT_RATIO
    ]
    expected = held_out_ratio(Split(SIZED, PARTS.test, 0), edges=EDGES)
    assert ratio == expected
    assert ratio > held_out_ratio(PARTS, edges=EDGES)


def test_fraction_one_keeps_e1s_ratio_bitwise(caplog):
    with caplog.at_level(logging.INFO):
        ready = prepared(PARTS, E1, edges=EDGES)
    assert ready.sized is PARTS
    assert ready.ablated is PARTS
    assert held_out_ratio(ready.sized, edges=EDGES) == held_out_ratio(PARTS, edges=EDGES)
    assert not [m for m in caplog.messages if m.startswith("subsample ")]


def test_subsample_and_removal_are_exclusive():
    with pytest.raises(ValueError, match="cannot both thin"):
        dataclasses.replace(
            E1, training_fraction=Fraction(1, 2), removal=Removal(("<55", "female", 1), 0.5)
        )


@pytest.mark.parametrize("fraction", [Fraction(0), Fraction(3, 2)])
def test_a_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValueError, match="must be a Fraction in"):
        dataclasses.replace(E1, training_fraction=fraction)


def test_rehearsal_ranges_follow_the_subsample():
    found = safe_ranges(GRADED, HALF, frame=UCI_FRAME)
    trains = [
        subsampled(split(GRADED, seed=s, edges=EDGES).train, Fraction(1, 2), seed=s, edges=EDGES)
        for s in HALF.holdout_seeds
    ]
    for name in UCI_FRAME.numeric:
        low = max(float(t[name].min()) for t in trains)
        high = min(float(t[name].max()) for t in trains)
        assert found[name] == (low, high)


def test_the_flagged_cells_and_gap_request_read_the_subsample(caplog):
    with caplog.at_level(logging.INFO):
        scores = score_holdout(PARTS, HALF, FAST_ARMS, frame=UCI_FRAME)
    sized = flagged(SIZED, HALF, edges=EDGES)
    assert sized != flagged(PARTS.train, HALF, edges=EDGES)
    names = ",".join("/".join(map(str, cell)) for cell in sized)
    assert f"flagged holdout=0 cells={names}" in caplog.messages
    (found,) = [
        s.value
        for s in scores
        if s.arm == HELD_OUT_ARM and s.measure is Measure.FLAGGED_FINGERPRINT
    ]
    assert found == fingerprint(sized)
    quota = gap_request(SIZED, HALF, seed=0, frame=UCI_FRAME).quota
    lines = [m for m in caplog.messages if m.startswith("quota ")]
    assert len(lines) == len(quota)


def test_the_likelihood_gate_reads_the_subsamples_ratio(monkeypatch):
    seen: list[float] = []
    real = likelihood_module.corrected_interval

    def spying(values, *, ratio, confidence, family):
        seen.append(ratio)
        return real(values, ratio=ratio, confidence=confidence, family=family)

    monkeypatch.setattr(likelihood_module, "corrected_interval", spying)
    lines = [
        candidate(seed, call, cell, p, row)
        for seed in HALF.holdout_seeds
        for call, (cell, rows) in enumerate(cells(seed), start=1)
        for p, row in enumerate(rows)
    ]
    likelihood_module.likelihood(HALF, DATA, [read(encode(lines))], io.StringIO())
    expected = max(
        held_out_ratio(
            prepared(split(GRADED, seed=s, edges=EDGES), HALF, edges=EDGES).sized, edges=EDGES
        )
        for s in HALF.holdout_seeds
    )
    assert seen
    assert set(seen) == {expected}
    assert expected > held_out_ratio(PARTS, edges=EDGES)
