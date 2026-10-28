"""The BKO arm: rounds, the budget refusal, drops, own-cell counting, validity and the ceiling."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from pathlib import Path
from typing import Final

import anthropic
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.evaluate.errors import EvaluationError, StopRuleError
from bko.evaluate.guided import SEQUENTIAL, GuidedArm, Knowledge, guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.evaluate.scores import Source
from bko.evaluate.validity import COUNTS
from bko.gaps.definitions import Definition
from bko.generate.api import async_send_with
from bko.generate.calls import ask
from bko.generate.errors import GenerationError, IdleError, ReplayError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AsyncSend, Call, Reply
from bko.generate.replay import Replay, Resume
from bko.generate.spend import cost
from bko.ontology.mapping import RELEASES
from tests._evaluate_fixtures import GRADED_DATA, SMALL
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply, request, response
from tests._guided_fixtures import REQUEST, YOUNG_SICK, guided, train
from tests._validate_fixtures import planted


ORIGIN: Final = Origin("e1", GRADED_DATA, RELEASES)
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


def test_guided_arms_render_each_holdouts_bounds(tmp_path):
    first = train(tmp_path / "train")
    second = first.copy()
    second.loc[second.index[-1], "chol"] = 400
    low = first["chol"].min()
    assert first["chol"].max() == low < 400
    send = ScriptedSend(reply([young(1)] * 3), reply([young(1)] * 3))
    arm, log = guided(tmp_path, send)
    with log:
        arm(first, REQUEST)
        arm(second, REQUEST)
    users = [call.user for call in send.calls]
    assert len(users) == 2
    assert f"chol {low:g} to {low:g} mg/dL" in users[0]
    assert f"chol {low:g} to 400 mg/dL" in users[1]


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


OLD_SICK: Final = (">=55", "female", 1)
"""`YOUNG_SICK`'s older band: the same sex and label, so a reply may keep a record in either."""


def _old_sick() -> dict[str, object]:
    return planted(age=63, sex=0, target=1)


def test_rows_come_out_cell_by_cell_in_the_quotas_order(tmp_path):
    request = Request(quota={YOUNG_SICK: 1, OLD_SICK: 2}, strata=Strata.CELL, seed=0)
    send = ScriptedSend(reply([_old_sick(), young(1)]), reply([_old_sick()]))
    arm, log = guided(tmp_path, send)
    with log:
        baseline = arm(train(tmp_path / "train"), request)
    assert asked(log.path) == [1, 1]
    assert baseline.added["age"].tolist() == [40, 63, 63]


def test_a_cell_filled_by_another_cells_call_gets_no_call(tmp_path):
    request = Request(quota={YOUNG_SICK: 1, OLD_SICK: 1}, strata=Strata.CELL, seed=0)
    send = ScriptedSend(reply([young(1), _old_sick()]))
    arm, log = guided(tmp_path, send)
    with log:
        baseline = arm(train(tmp_path / "train"), request)
    send.assert_consumed()
    assert asked(log.path) == [1]
    assert baseline.added["age"].tolist() == [40, 63]


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
            run="e1",
            limits=SEQUENTIAL,
            answers=resume,
        ),
    ):
        pass
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["status"] == "failed"
    assert "validity arm=bko" in caplog.text


def test_an_arm_needs_the_runs_flight(tmp_path):
    """An arm building its own `Flight` would give each wave a fresh event loop, which an
    async client's connections, bound to the first, cannot use."""
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS)
    with pytest.raises(TypeError, match="flight"):
        GuidedArm(  # ty: ignore[missing-argument]
            "bko", Definition.SUBGROUP, Knowledge("x", {}, (), ()), log=log, budget=3
        )


def test_each_call_of_an_arm_runs_under_the_idle_rule(tmp_path):
    def answered(call):
        async def land():
            return reply([young(1)] * 3)

        return land()

    arm, log = guided(tmp_path, answered, idle=5.0)
    with log:
        arm(train(tmp_path / "train"), REQUEST)
    assert asked(log.path) == [3]


def test_a_stalled_call_stops_the_run_and_the_log_ends_failed(tmp_path, caplog):
    def stalled(call):
        return asyncio.Event().wait()

    arm, log = guided(tmp_path, stalled, idle=0.5)
    caplog.set_level(logging.WARNING)
    with pytest.raises(StopRuleError, match="idle limit") as raised, log:
        arm(train(tmp_path / "train"), REQUEST)
    assert raised.value.stop_rule == "idle"
    assert raised.value.retryable is True
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["status"] == "failed"
    assert json.loads(lines[-1])["stop"] == "idle"
    assert json.loads(lines[-1])["retryable"] is True
    assert "stop rule=idle call=1 limit_seconds=0.5" in caplog.text


def test_guided_arms_projects_a_live_run_and_a_resumed_one_but_not_a_replay(tmp_path):
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
        run="e1",
        limits=SEQUENTIAL,
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
        run="e1",
        limits=SEQUENTIAL,
        answers=resume,
    ) as resumed:
        assert resumed.watch.projecting
    with guided_arms(
        ("bko",),
        knowledge,
        early,
        log=tmp_path / "replayed.jsonl",
        candidates=tmp_path / "replayed-candidates.jsonl",
        origin=ORIGIN,
        send=ScriptedSend,
        run="e1",
        limits=SEQUENTIAL,
        answers=Replay(stopped),
    ) as replayed:
        assert not replayed.watch.projecting


def test_a_resume_projects_the_whole_runs_spend_over_every_holdout(tmp_path, caplog):
    """A segment stopped before `projection_after` holdouts leaves its resume projecting: the
    whole run's spend, the replayed calls' included, over every holdout done."""
    stopped = tmp_path / "stopped.jsonl"
    with contextlib.suppress(StopRuleError), CallLog(stopped, SETTINGS) as log:
        ask(GUIDANCE, 2, send=ScriptedSend(reply([young(1), young(1)])), log=log)
        raise StopRuleError("stalled", "idle", retryable=True)
    one = json.loads(stopped.read_text(encoding="utf-8").splitlines()[-1])["usd"]
    assert one > 0
    resume = Resume(stopped, ScriptedSend())
    early = dataclasses.replace(SMALL, projection_after=1, spend_ceiling=one * 1.5)

    def send(call: Call) -> Reply:
        answer = resume(call)
        assert isinstance(answer, Reply)
        return answer

    def run() -> None:
        with guided_arms(
            ("bko",),
            Knowledge("x", {}, (), ()),
            early,
            log=tmp_path / "resumed.jsonl",
            candidates=tmp_path / "resumed-candidates.jsonl",
            origin=ORIGIN,
            send=ScriptedSend,
            run="e1",
            limits=SEQUENTIAL,
            answers=resume,
        ) as resumed:
            ask(GUIDANCE, 2, send=send, log=resumed.log)
            assert resumed.log.live_usd == 0
            resumed.watch.holdout_done()

    caplog.set_level(logging.INFO)
    with pytest.raises(StopRuleError, match="projects to") as raised:
        run()
    assert raised.value.stop_rule == "projection"
    assert f"projection holdouts=1 of=2 usd={one:.4f} projected_usd={2 * one:.4f}" in caplog.text
    end = json.loads((tmp_path / "resumed.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert (end["status"], end["stop"]) == ("failed", "projection")


def _sdk_send(monkeypatch: pytest.MonkeyPatch, exc: anthropic.APIError) -> AsyncSend:
    """The async send over a real client whose `messages.create` raises `exc`."""
    client = anthropic.AsyncAnthropic(api_key="not-a-credential", max_retries=0)

    async def create(**_: object) -> anthropic.types.Message:
        raise exc

    monkeypatch.setattr(client.messages, "create", create)
    return async_send_with(client)


SDK_FAILURES: Final = {
    "connection": (lambda: anthropic.APIConnectionError(request=request()), True),
    "timeout": (lambda: anthropic.APITimeoutError(request=request()), True),
    "rate-limit": (lambda: anthropic.RateLimitError("x", response=response(429), body=None), True),
    "server": (
        lambda: anthropic.InternalServerError("x", response=response(500), body=None),
        True,
    ),
    "overloaded": (lambda: anthropic.APIStatusError("x", response=response(529), body=None), True),
    "bad-request": (
        lambda: anthropic.BadRequestError("x", response=response(400), body=None),
        False,
    ),
    "unreadable": (lambda: anthropic.APIResponseValidationError(response(200), None), False),
}
"""Each SDK failure a call can meet, and whether the stop it causes is the API's to retry."""


@pytest.mark.parametrize("name", SDK_FAILURES)
def test_the_end_line_marks_api_failures_retryable(tmp_path, monkeypatch, name):
    build, retryable = SDK_FAILURES[name]
    arm, log = guided(tmp_path, _sdk_send(monkeypatch, build()))
    with pytest.raises(GenerationError), log:
        arm(train(tmp_path / "train"), REQUEST)
    end = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert (end["status"], end["retryable"]) == ("failed", retryable)


@pytest.mark.parametrize(
    ("raised", "retryable"),
    [
        (StopRuleError("stalled", "idle", retryable=True), True),
        (StopRuleError("stalled", "idle"), False),
        (IdleError("call 1 got no reply"), True),
        (StopRuleError("spent", "ceiling"), False),
        (StopRuleError("projected", "projection"), False),
        (GenerationError("no Anthropic credential"), False),
        (BaselineError("refused"), False),
        (ValueError("bug"), False),
    ],
    ids=[
        "idle",
        "idle-unmarked",
        "idle-error",
        "ceiling",
        "projection",
        "generation",
        "baseline",
        "other",
    ],
)
def test_the_end_line_reads_retryable_from_the_error_it_ended_on(tmp_path, raised, retryable):
    path = tmp_path / "calls.jsonl"
    with pytest.raises(type(raised)), CallLog(path, SETTINGS):
        raise raised
    end = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert (end["status"], end["retryable"]) == ("failed", retryable)


@pytest.mark.parametrize("raised", [None, KeyboardInterrupt()])
def test_a_complete_or_interrupted_run_is_not_retryable(tmp_path, raised):
    path = tmp_path / "calls.jsonl"
    with contextlib.suppress(KeyboardInterrupt), CallLog(path, SETTINGS):
        if raised is not None:
            raise raised
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["retryable"] is False
