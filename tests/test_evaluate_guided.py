"""The BKO arm: rounds, the budget refusal, drops, own-cell counting, validity and the ceiling."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Final

import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.evaluate.errors import EvaluationError
from bko.evaluate.guided import GuidedArm, Knowledge, guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.evaluate.scores import Source
from bko.evaluate.validity import COUNTS
from bko.evaluate.watch import Watch
from bko.gaps.definitions import Definition
from bko.generate.calls import ask
from bko.generate.errors import GenerationError, ReplayError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS
from bko.generate.replay import Resume
from bko.generate.spend import cost
from bko.ontology.mapping import RELEASES
from tests._evaluate_fixtures import SMALL, Timers, interrupt
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply
from tests._guided_fixtures import REQUEST, YOUNG_SICK, YOUNG_WELL, guided, train
from tests._validate_fixtures import planted


ORIGIN: Final = Origin("e1", UCI_HEART_DISEASE, RELEASES)
"""The header of every candidates file these tests make."""

OLD: Final = planted(age=63, sex=0, target=1)
"""A valid candidate of `(">=55", "female", 1)`, a cell `REQUEST` does not ask for."""


def asked(path: Path) -> list[int]:
    """The rows each logged call asked for, in call order."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [line["asked"] for line in lines if line["kind"] == "request"]


def counted(arm: GuidedArm) -> tuple[float, ...]:
    """The arm's drained counts: generated, passed, dropped for range, cell and quota."""
    found = arm.drain(0, Source.CANDIDATES)
    return tuple(found[measure] for measure in COUNTS)


@pytest.mark.parametrize("definition", list(Definition))
def test_a_request_is_filled_from_valid_candidates(tmp_path, definition):
    arm, log = guided(tmp_path, ScriptedSend(reply([young(1)] * 3)), definition=definition)
    with log:
        baseline = arm(train(tmp_path / "train"), REQUEST)
    assert list(baseline.added.index) == [-1, -2, -3]
    assert asked(log.path) == [3]
    assert counted(arm) == pytest.approx((3, 3, 0, 0, 0))
    assert counted(arm) == pytest.approx((0, 0, 0, 0, 0))


def test_each_reason_is_dropped_and_the_shortfall_asked_again(tmp_path):
    first = reply([young(1, chol=234), OLD, young(1), young(1)])
    arm, log = guided(tmp_path, ScriptedSend(first, reply([young(1), young(1)])))
    with log:
        baseline = arm(train(tmp_path / "train"), REQUEST)
    assert asked(log.path) == [3, 1]
    assert counted(arm) == pytest.approx((6, 6, 1, 1, 1))
    assert len(baseline.added) == 3


def test_the_budget_refuses_with_a_plain_baseline_error(tmp_path, caplog):
    send = ScriptedSend(reply([young(1, chol=234)] * 3))
    arm, log = guided(tmp_path, send, budget=1)
    caplog.set_level(logging.INFO, logger="bko.evaluate.guided")
    with log, pytest.raises(BaselineError) as raised:
        arm(train(tmp_path / "train"), REQUEST)
    line = "asked=3 generated=3 passed=3 dropped_range=3 dropped_cell=0 dropped_quota=0 kept=0"
    assert type(raised.value) is BaselineError
    assert line in str(raised.value)
    assert any("outcome=refused" in r.message and line in r.message for r in caplog.records)
    send.assert_consumed()


def test_a_refused_attempt_stays_in_the_counts(tmp_path):
    send = ScriptedSend(reply([young(1, chol=234)] * 3), reply([young(1)] * 3))
    arm, log = guided(tmp_path, send, budget=1)
    rows = train(tmp_path / "train")
    with log:
        with pytest.raises(BaselineError):
            arm(rows, REQUEST)
        arm(rows, REQUEST)
    assert counted(arm) == pytest.approx((6, 6, 3, 0, 0))


def test_an_undecoded_record_counts_as_generated_and_not_passed(tmp_path):
    arm, log = guided(tmp_path, ScriptedSend(reply([], stop_reason="max_tokens")), budget=1)
    with log, pytest.raises(BaselineError):
        arm(train(tmp_path / "train"), REQUEST)
    assert counted(arm)[:2] == pytest.approx((3, 0))


def test_a_call_with_no_response_is_not_a_refusal(tmp_path):
    failure = GenerationError("the API could not be reached: APIConnectionError")
    arm, log = guided(tmp_path, ScriptedSend(failure))
    with pytest.raises(GenerationError), log:
        arm(train(tmp_path / "train"), REQUEST)
    assert counted(arm)[0] == pytest.approx(0)


def test_the_run_stops_at_its_spend_ceiling(tmp_path):
    first = reply([young(1)])
    arm, log = guided(tmp_path, ScriptedSend(first), ceiling=cost(first.usage, SETTINGS.model))
    with pytest.raises(EvaluationError, match="ceiling"), log:
        arm(train(tmp_path / "train"), REQUEST)
    assert asked(log.path) == [3]


def test_the_ceiling_is_read_before_every_call_of_a_cell(tmp_path, caplog):
    first = reply([young(1)] * SETTINGS.per_call)
    spent = cost(first.usage, SETTINGS.model)
    arm, log = guided(tmp_path, ScriptedSend(first), ceiling=spent)
    request = Request(quota={YOUNG_SICK: SETTINGS.per_call + 1}, strata=Strata.CELL, seed=0)
    caplog.set_level(logging.INFO, logger="bko.evaluate.guided")
    with pytest.raises(EvaluationError, match="ceiling"), log:
        arm(train(tmp_path / "train"), request)
    assert asked(log.path) == [SETTINGS.per_call]
    spends = [m for m in caplog.messages if m.startswith("spend ")]
    assert spends == [
        f"spend usd=0.0000 ceiling={spent:.2f}",
        f"spend usd={spent:.4f} ceiling={spent:.2f}",
    ]


def test_rows_come_out_cell_by_cell_in_the_quotas_order(tmp_path):
    request = Request(quota={YOUNG_WELL: 1, YOUNG_SICK: 2}, strata=Strata.CELL, seed=0)
    send = ScriptedSend(reply([young(1), young(0)]), reply([young(1)]))
    arm, log = guided(tmp_path, send)
    with log:
        baseline = arm(train(tmp_path / "train"), request)
    assert asked(log.path) == [1, 1]
    assert baseline.added["target"].tolist() == [0, 1, 1]


def test_a_cell_filled_by_another_cells_call_gets_no_call(tmp_path):
    request = Request(quota={YOUNG_WELL: 1, YOUNG_SICK: 1}, strata=Strata.CELL, seed=0)
    send = ScriptedSend(reply([young(0), young(1)]))
    arm, log = guided(tmp_path, send)
    with log:
        baseline = arm(train(tmp_path / "train"), request)
    send.assert_consumed()
    assert asked(log.path) == [1]
    assert baseline.added["target"].tolist() == [0, 1]


def test_a_label_request_is_refused(tmp_path):
    arm, log = guided(tmp_path, ScriptedSend())
    with pytest.raises(EvaluationError, match="cells only"), log:
        arm(train(tmp_path / "train"), Request(quota={(1,): 3}, strata=Strata.LABEL))


def test_the_summary_logs_the_runs_sums(tmp_path, caplog):
    arm, log = guided(tmp_path, ScriptedSend(reply([young(1)] * 3)))
    with log:
        arm(train(tmp_path / "train"), REQUEST)
    caplog.set_level(logging.INFO, logger="bko.evaluate.guided")
    arm.summary()
    sums = "validity arm=bko generated=3 passed=3 dropped_range=0 dropped_cell=0 dropped_quota=0"
    assert sums in caplog.text


def test_guided_arms_refuses_a_resume_with_answers_left_and_logs_the_sums(tmp_path, caplog):
    stopped = tmp_path / "stopped.jsonl"
    with CallLog(stopped, SETTINGS) as log:
        ask(GUIDANCE, 2, send=ScriptedSend(reply([young(1), young(1)])), log=log)
    resume = Resume(stopped, ScriptedSend())
    knowledge = Knowledge("x", {}, (), ())
    path = tmp_path / "resumed.jsonl"
    caplog.set_level(logging.INFO)
    with (
        pytest.raises(ReplayError, match="1 logged calls were not replayed"),
        guided_arms(
            ("bko",),
            knowledge,
            SMALL,
            log=path,
            candidates=tmp_path / "candidates.jsonl",
            origin=ORIGIN,
            send=ScriptedSend,
            answers=resume,
        ),
    ):
        pass
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["status"] == "failed"
    assert "validity arm=bko" in caplog.text


def test_each_call_of_an_arm_runs_under_the_idle_rule(tmp_path):
    timers = Timers()
    watch = Watch(SMALL, lambda: 0.0, timers=timers, interrupt=interrupt)
    arm, log = guided(tmp_path, ScriptedSend(reply([young(1)] * 3)), watch=watch)
    with log:
        arm(train(tmp_path / "train"), REQUEST)
    assert asked(log.path) == [3]
    assert [(t.started, t.cancelled) for t in timers.made] == [(True, True)]


def test_a_stalled_call_stops_the_run_and_the_log_ends_failed(tmp_path, caplog):
    timers = Timers()
    watch = Watch(SMALL, lambda: 0.0, timers=timers, interrupt=interrupt)

    def stalled(call):
        timers.made[-1].action()

    arm, log = guided(tmp_path, stalled, watch=watch)
    caplog.set_level(logging.WARNING)
    with pytest.raises(EvaluationError, match="idle limit"), log:
        arm(train(tmp_path / "train"), REQUEST)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["status"] == "failed"
    assert "stop rule=idle" in caplog.text


def test_guided_arms_projects_a_live_run_and_not_a_resumed_one(tmp_path):
    knowledge = Knowledge("x", {}, (), ())
    early = dataclasses.replace(SMALL, projection_after=1)
    with guided_arms(
        ("bko",),
        knowledge,
        early,
        log=tmp_path / "live.jsonl",
        candidates=tmp_path / "live-candidates.jsonl",
        origin=ORIGIN,
        send=ScriptedSend,
    ) as live:
        assert live.watch.projecting
    stopped = tmp_path / "stopped.jsonl"
    with CallLog(stopped, SETTINGS):
        pass
    resume = Resume(stopped, ScriptedSend())
    with guided_arms(
        ("bko",),
        knowledge,
        early,
        log=tmp_path / "resumed.jsonl",
        candidates=tmp_path / "resumed-candidates.jsonl",
        origin=ORIGIN,
        send=ScriptedSend,
        answers=resume,
    ) as resumed:
        assert not resumed.watch.projecting
