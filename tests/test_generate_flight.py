"""`bko.generate.flight`: a wave's calls go out in plan order and land in any order."""

from __future__ import annotations

import asyncio
import json

import pytest

from bko.generate.errors import GenerationError
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import GUIDANCE, OTHER_GUIDANCE, reply


def _log(tmp_path) -> CallLog:
    return CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW)


def _lines(log: CallLog) -> list[dict[str, object]]:
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


def _asked(call: Call) -> int:
    return int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])


PLAN = (Planned(GUIDANCE, 1), Planned(OTHER_GUIDANCE, 2), Planned(GUIDANCE, 3))
"""Three calls asking 1, 2 and 3 records, so each reply and request line names its call."""


class Delayed:
    """An async send answering each call in full after a delay that shrinks with its number.

    Attributes:
        calls: Every call made, in order.
        landed: The call numbers in the order their replies landed.
    """

    def __init__(self, *, fail: dict[int, BaseException] | None = None) -> None:
        self.calls: list[Call] = []
        self.landed: list[int] = []
        self._fail = fail or {}

    def __call__(self, call: Call):
        self.calls.append(call)
        number = len(self.calls)

        async def land() -> Reply:
            await asyncio.sleep(0.03 / number)
            self.landed.append(number)
            if number in self._fail:
                raise self._fail[number]
            return reply([{"age": 40}] * _asked(call))

        return land()


def test_a_wave_requests_in_plan_order_and_returns_in_plan_order(tmp_path):
    send = Delayed()
    checks: list[int] = []
    with (
        _log(tmp_path) as log,
        Flight(
            send, log, Limits(in_flight=3, idle=None), spendable=lambda: checks.append(log.calls)
        ) as flight,
    ):
        batches = flight.wave(PLAN)
    assert send.landed == [3, 2, 1]
    assert checks == [0, 1, 2]
    assert [batch.index for batch in batches] == [1, 2, 3]
    assert [batch.asked for batch in batches] == [1, 2, 3]
    assert [batch.cell for batch in batches] == [p.guidance.cell for p in PLAN]
    lines = _lines(log)
    assert [line["asked"] for line in lines if line["kind"] == "request"] == [1, 2, 3]
    assert [line["call"] for line in lines if line["kind"] == "reply"] == [3, 2, 1]


def test_a_wave_at_in_flight_1_sends_each_call_after_the_one_before_lands(tmp_path):
    send = Delayed()
    with (
        _log(tmp_path) as log,
        Flight(send, log, Limits(in_flight=1, idle=None), spendable=lambda: None) as flight,
    ):
        flight.wave(PLAN)
    assert send.landed == [1, 2, 3]
    kinds = [line["kind"] for line in _lines(log)[1:-1]]
    assert kinds == ["request", "reply"] * 3


def test_a_failure_raises_the_lowest_failing_index(tmp_path):
    second = GenerationError("call two failed")
    third = GenerationError("call three failed")
    send = Delayed(fail={2: second, 3: third})
    log = _log(tmp_path)
    with pytest.raises(GenerationError) as caught, log:
        Flight(send, log, Limits(in_flight=3, idle=None), spendable=lambda: None).wave(PLAN)
    assert caught.value is second
    lines = _lines(log)
    assert {line["call"]: line["error"] for line in lines if line["kind"] == "error"} == {
        2: "call two failed",
        3: "call three failed",
    }
    assert [line["call"] for line in lines if line["kind"] == "reply"] == [1]


def test_a_failure_at_in_flight_1_sends_no_later_call(tmp_path):
    send = Delayed(fail={1: GenerationError("no response")})
    log = _log(tmp_path)
    with pytest.raises(GenerationError), log:
        Flight(send, log, Limits(in_flight=1, idle=None), spendable=lambda: None).wave(PLAN)
    assert len(send.calls) == 1
    assert [line["kind"] for line in _lines(log)] == ["run", "request", "error", "end"]


def test_a_keyboard_interrupt_mid_wave_ends_interrupted(tmp_path):
    send = Delayed(fail={2: KeyboardInterrupt()})
    log = _log(tmp_path)
    with (
        pytest.raises(KeyboardInterrupt),
        log,
        Flight(send, log, Limits(in_flight=3, idle=None), spendable=lambda: None) as flight,
    ):
        flight.wave(PLAN)
    lines = _lines(log)
    assert lines[-1]["status"] == "interrupted"
    answered = {line["call"]: line["kind"] for line in lines if line["kind"] in {"reply", "error"}}
    assert answered == {1: "error", 2: "error", 3: "reply"}
    assert {line["call"]: line["error"] for line in lines if line["kind"] == "error"} == {
        1: "CancelledError",
        2: "KeyboardInterrupt",
    }


def test_two_live_sends_overlap(tmp_path):
    events = (asyncio.Event(), asyncio.Event())
    calls: list[Call] = []

    def send(call: Call):
        calls.append(call)
        mine, theirs = (events[0], events[1]) if len(calls) == 1 else (events[1], events[0])

        async def meet() -> Reply:
            mine.set()
            async with asyncio.timeout(2):
                await theirs.wait()
            return reply([{"age": 40}] * _asked(call))

        return meet()

    with (
        _log(tmp_path) as log,
        Flight(send, log, Limits(in_flight=2, idle=None), spendable=lambda: None) as flight,
    ):
        batches = flight.wave(PLAN[:2])
    assert [batch.index for batch in batches] == [1, 2]


def test_a_flight_outside_a_with_block_runs_each_wave_on_its_own_loop(tmp_path):
    send = Delayed()
    with _log(tmp_path) as log:
        flight = Flight(send, log, Limits(in_flight=2, idle=None), spendable=lambda: None)
        assert [batch.index for batch in flight.wave(PLAN[:1])] == [1]
        assert [batch.index for batch in flight.wave(PLAN[1:])] == [2, 3]


@pytest.mark.parametrize("in_flight", [0, -1])
def test_limits_refuse_fewer_than_one_call_in_flight(in_flight):
    with pytest.raises(ValueError, match="in_flight"):
        Limits(in_flight=in_flight, idle=None)


def test_a_skeleton_is_refused_until_it_is_typed(tmp_path):
    with pytest.raises(ValueError, match="skeleton"), _log(tmp_path) as log:
        Flight(Delayed(), log, Limits(1, None), spendable=lambda: None).wave(
            (Planned(GUIDANCE, 1, skeleton=object()),)
        )
