"""The paid run's stop rules: each fires on its own condition, with its logged reason."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from bko.evaluate.errors import EvaluationError
from bko.evaluate.run import run
from bko.evaluate.watch import Watch
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

RULES = dataclasses.replace(
    SMALL,
    holdout_seeds=tuple(range(10)),
    projection_after=2,
    spend_ceiling=5.0,
    idle_limit=60.0,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_spend_projecting_past_the_ceiling_stops_the_run_with_its_measure(caplog):
    spent = [0.0]
    watch = Watch(RULES, lambda: spent[0])
    caplog.set_level(logging.INFO)
    watch.holdout_done()
    spent[0] = 1.2
    with pytest.raises(EvaluationError, match=r"projects to 6\.0000 USD over 10"):
        watch.holdout_done()
    assert "stop rule=projection projected_usd=6.0000 ceiling=5.00" in caplog.text


def test_a_projection_within_the_ceiling_never_fires(caplog):
    watch = Watch(RULES, lambda: 1.0)
    caplog.set_level(logging.INFO)
    for _ in range(10):
        watch.holdout_done()
    assert "projection holdouts=2 of=10 usd=1.0000 projected_usd=5.0000" in caplog.text
    assert "stop rule" not in caplog.text


def test_the_projection_is_read_once_after_the_configured_holdouts():
    watch = Watch(RULES, lambda: 1.2)
    watch.holdout_done()
    with pytest.raises(EvaluationError):
        watch.holdout_done()
    # The run is already stopped by then; a caller that goes on is not read again.
    watch.holdout_done()


def test_a_replay_is_never_projected():
    watch = Watch(RULES, lambda: 1000.0, projecting=False)
    for _ in range(10):
        watch.holdout_done()


def test_a_projection_after_every_holdout_has_nothing_to_scale():
    late = dataclasses.replace(RULES, projection_after=10)
    watch = Watch(late, lambda: 1000.0)
    for _ in range(10):
        watch.holdout_done()


def test_no_completed_call_within_the_idle_limit_stops_the_run_with_its_measure(caplog):
    clock = Clock()
    watch = Watch(RULES, lambda: 0.0, clock=clock)
    caplog.set_level(logging.INFO)
    clock.now += 60
    watch.before_call()
    clock.now += 1
    with pytest.raises(EvaluationError, match="no call completed in 61 s, past the idle limit"):
        watch.before_call()
    assert "stop rule=idle idle_seconds=61.0 limit_seconds=60.0" in caplog.text


def test_a_completed_call_restarts_the_idle_clock():
    clock = Clock()
    watch = Watch(RULES, lambda: 0.0, clock=clock)
    for _ in range(5):
        clock.now += 50
        watch.before_call()
        watch.call_done()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"projection_after": 0}, "projection after 0 holdouts"),
        ({"idle_limit": 0.0}, "idle limit"),
    ],
)
def test_an_experiment_refuses_a_rule_that_cannot_fire(changes, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SMALL, **changes)


def test_the_harness_calls_on_holdout_after_each_holdout():
    light = dataclasses.replace(
        SMALL, arms=("none", "smote"), candidate="smote", comparators=(), replica_arms=()
    )
    calls: list[int] = []
    run(GRADED, light, FAST_ARMS, on_holdout=lambda: calls.append(len(calls)))
    assert calls == list(range(len(light.holdout_seeds)))
