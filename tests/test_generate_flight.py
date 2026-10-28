"""`bko.generate.flight`: a wave's calls go out in plan order and land in any order."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import time
from functools import partial
from typing import Final

import pytest

from bko.evaluate.errors import StopRuleError
from bko.evaluate.guided import spendable
from bko.generate.calls import request
from bko.generate.errors import GenerationError, IdleError
from bko.generate.flight import OVERHEAD_TOKENS, Flight, Limits, Planned, _reservation
from bko.generate.log import CallLog
from bko.generate.messages import ATTEMPTS, CALL_NUMBER, SETTINGS, Call, Reply, Usage
from bko.generate.skeleton import Skeleton
from bko.generate.spend import PRICES, cost
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import GUIDANCE, OTHER_GUIDANCE, reply


def _log(tmp_path) -> CallLog:
    return CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci")


def _lines(log: CallLog) -> list[dict[str, object]]:
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


def _asked(call: Call) -> int:
    return int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])


PLAN = (Planned(GUIDANCE, 1), Planned(OTHER_GUIDANCE, 2), Planned(GUIDANCE, 3))
"""Three calls asking 1, 2 and 3 records, so each reply and request line names its call."""


STEP: Final = 0.3
"""The first call's delay in seconds under `Delayed`; call n's is `STEP / n`, so the replies of
calls 1 to 3 land at least 50 ms apart, last first."""


class Delayed:
    """An async send answering each call in full after a delay that shrinks with its number.

    Attributes:
        calls: Every call made, in order.
        landed: The call numbers in the order their replies landed.
    """

    def __init__(
        self, *, fail: dict[int, BaseException] | None = None, usage: Usage | None = None
    ) -> None:
        self.calls: list[Call] = []
        self.landed: list[int] = []
        self._fail = fail or {}
        self._usage = usage

    def __call__(self, call: Call):
        self.calls.append(call)
        number = len(self.calls)

        async def land() -> Reply:
            await asyncio.sleep(STEP / number)
            self.landed.append(number)
            if number in self._fail:
                raise self._fail[number]
            return reply([{"age": 40}] * _asked(call), usage=self._usage)

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


def test_a_failure_at_in_flight_1_sends_no_later_call(tmp_path, caplog):
    send = Delayed(fail={1: GenerationError("no response")})
    log = _log(tmp_path)
    caplog.set_level(logging.INFO)
    with pytest.raises(GenerationError), log:
        Flight(send, log, Limits(in_flight=1, idle=None), spendable=lambda: None).wave(PLAN)
    assert len(send.calls) == 1
    assert [line["kind"] for line in _lines(log)] == ["run", "request", "error", "end"]
    assert [m for m in caplog.messages if m.startswith("halt ")] == ["halt call=2 unsent=2"]


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


def test_a_skeleton_that_does_not_hold_a_line_per_record_is_refused(tmp_path):
    lines = Skeleton((), (frozenset(), frozenset()))
    with _log(tmp_path) as log:
        flight = Flight(Delayed(), log, Limits(1, None), spendable=lambda: None)
        with pytest.raises(ValueError, match="2 lines for 1 records"):
            flight.wave((Planned(GUIDANCE, 1, skeleton=lines),))
        assert log.calls == 0


SPENT: Final = cost(reply([]).usage, SETTINGS.model)
"""What each of `Delayed`'s replies costs."""

FULL: Final = Usage(1000, SETTINGS.max_tokens, 0, 0)
"""A reply's usage at its whole output budget, which its call's reservation still covers."""

FULL_USD: Final = cost(FULL, SETTINGS.model)


def _ceiling_run(tmp_path, in_flight: int, ceiling: float, caplog) -> tuple[Delayed, list[str]]:
    """Send `PLAN` under `ceiling` at `in_flight`; return the send and the stop rule lines."""
    send = Delayed(usage=FULL)
    log = CallLog(
        tmp_path / f"calls-{in_flight}.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci"
    )
    caplog.clear()
    limits = Limits(in_flight=in_flight, idle=None)
    with (
        pytest.raises(StopRuleError, match="reached its ceiling") as raised,
        log,
        Flight(send, log, limits, spendable=partial(spendable, log, ceiling), ceiling=ceiling) as f,
    ):
        f.wave(PLAN)
    assert raised.value.stop_rule == "ceiling"
    return send, [m for m in caplog.messages if m.startswith("stop rule=")]


def test_a_mid_wave_ceiling_stops_at_the_in_flight_1_call(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    ceiling = 2 * FULL_USD
    assert f"{ceiling:.2f}" != "0.00"
    one, stop_one = _ceiling_run(tmp_path, 1, ceiling, caplog)
    many, stop_many = _ceiling_run(tmp_path, 3, ceiling, caplog)
    assert len(one.calls) == len(many.calls) == 2
    assert stop_many == stop_one == [f"stop rule=ceiling usd={ceiling:.4f} ceiling={ceiling:.2f}"]
    waits = [m for m in caplog.messages if m.startswith("ceiling wait ")]
    assert len(waits) == 1
    assert re.fullmatch(
        rf"ceiling wait call=3 reserved_usd=\d+\.\d{{4}} in_flight=2 usd=0\.0000 "
        rf"ceiling={ceiling:.2f}",
        waits[0],
    ), waits
    lines = [json.loads(line) for line in (tmp_path / "calls-3.jsonl").read_text().splitlines()]
    assert lines[-1]["calls"] == 2
    assert lines[-1]["stop"] == "ceiling"


def test_a_reservation_prices_its_input_at_the_cache_write_rate(tmp_path):
    price = PRICES[SETTINGS.model]
    assert price.cache_write > price.input
    with _log(tmp_path) as log:
        pending = request(GUIDANCE, 1, log=log)
    text = len(json.dumps(pending.call.params()))
    expected = (
        SETTINGS.max_tokens * price.output + (text + OVERHEAD_TOKENS) * price.cache_write
    ) / 1e6
    assert _reservation(pending) == pytest.approx(expected)


def test_a_ceiling_the_reservations_stay_under_keeps_every_call_in_flight(tmp_path):
    send = Delayed()
    with (
        _log(tmp_path) as log,
        Flight(
            send,
            log,
            Limits(in_flight=3, idle=None),
            spendable=partial(spendable, log, 1.0),
            ceiling=1.0,
        ) as flight,
    ):
        flight.wave(PLAN)
    assert send.landed == [3, 2, 1]


def test_a_call_that_fails_while_the_ceiling_waits_stops_the_wave(tmp_path):
    send = Delayed(fail={1: GenerationError("no response")})
    log = _log(tmp_path)
    limits = Limits(in_flight=3, idle=None)
    with pytest.raises(GenerationError), log:
        Flight(send, log, limits, spendable=partial(spendable, log, SPENT), ceiling=SPENT).wave(
            PLAN
        )
    assert len(send.calls) == 1


def test_a_ceiling_stop_outranks_a_call_in_flight_that_fails(tmp_path):
    send = Delayed(fail={1: GenerationError("no response")})
    log = _log(tmp_path)

    def refuse_the_third() -> None:
        if log.calls == 2:
            msg = "the run's spend reached its ceiling"
            raise StopRuleError(msg, "ceiling")

    flight = Flight(send, log, Limits(in_flight=3, idle=None), spendable=refuse_the_third)
    with pytest.raises(StopRuleError) as raised, log:
        flight.wave(PLAN)
    assert raised.value.stop_rule == "ceiling"
    lines = _lines(log)
    assert [line["call"] for line in lines if line["kind"] == "error"] == [1]
    assert [line["call"] for line in lines if line["kind"] == "reply"] == [2]
    assert lines[-1]["stop"] == "ceiling"


class Stalled(Delayed):
    """`Delayed`, except that calls 1 and 3 never land."""

    def __call__(self, call: Call):
        sent = super().__call__(call)
        if len(self.calls) == 2:
            return sent
        sent.close()

        async def never() -> Reply:
            await asyncio.Event().wait()
            raise AssertionError(call)

        return never()


def test_an_idle_send_stops_with_idle_and_no_signal(tmp_path, monkeypatch, caplog):
    def no_signal(*_: object) -> None:
        raise AssertionError("the idle rule sent a signal")

    monkeypatch.setattr(signal, "pthread_kill", no_signal)
    log = _log(tmp_path)
    caplog.set_level(logging.WARNING)
    with (
        pytest.raises(IdleError, match="call 1 got no reply within the idle limit") as raised,
        log,
        Flight(Stalled(), log, Limits(in_flight=3, idle=0.5), spendable=lambda: None) as f,
    ):
        f.wave(PLAN)
    assert raised.value.stop_rule == "idle"
    lines = _lines(log)
    errors = [(line["call"], line["error"]) for line in lines if line["kind"] == "error"]
    # Call 3 was in flight beside call 1: it lands under its own idle limit, not cancelled.
    assert errors == [
        (1, str(raised.value)),
        (3, "call 3 got no reply within the idle limit of 0.5 s"),
    ]
    assert [line["call"] for line in lines if line["kind"] == "reply"] == [2]
    assert lines[-1]["status"] == "failed"
    assert lines[-1]["stop"] == "idle"
    assert [m for m in caplog.messages if m.startswith("stop rule=")] == [
        "stop rule=idle call=1 limit_seconds=0.5"
    ]


def _answered_lines(log: CallLog) -> dict[object, list[str]]:
    """Each requested call's answer line kinds, by call number."""
    lines = _lines(log)
    answers: dict[object, list[str]] = {
        line["call"]: [] for line in lines if line["kind"] == "request"
    }
    for line in lines:
        if line["kind"] in {"reply", "error"}:
            answers[line["call"]].append(str(line["kind"]))
    return answers


def test_a_call_beside_an_idle_stop_lands_and_its_spend_is_counted(tmp_path):
    log = _log(tmp_path)

    def send(call: Call):
        number = _asked(call)

        async def land() -> Reply:
            if number == 1:
                await asyncio.Event().wait()
            if number == 2:
                await asyncio.sleep(0.2)
            # Call 3 replies once call 1's idle stop is written, inside its own idle limit.
            while number == 3 and log.failed == 0:
                await asyncio.sleep(0.01)
            return reply([{"age": 40}] * number)

        return land()

    with (
        pytest.raises(IdleError, match="call 1 got no reply"),
        log,
        Flight(send, log, Limits(in_flight=2, idle=0.5), spendable=lambda: None) as f,
    ):
        f.wave(PLAN)
    assert _answered_lines(log) == {1: ["error"], 2: ["reply"], 3: ["reply"]}
    assert log.usd == pytest.approx(2 * SPENT)


def test_an_idle_calls_error_line_counts_its_attempts_and_rate_limits(tmp_path):
    def send(call: Call):
        async def stall() -> Reply:
            holder = ATTEMPTS.get()
            assert holder is not None
            holder.count = 3
            holder.rate_limited = True
            await asyncio.Event().wait()
            raise AssertionError(call)

        return stall()

    log = _log(tmp_path)
    with (
        pytest.raises(IdleError) as raised,
        log,
        Flight(send, log, Limits(in_flight=1, idle=0.5), spendable=lambda: None) as f,
    ):
        f.wave(PLAN[:1])
    assert raised.value.attempts == 3
    assert raised.value.rate_limited
    lines = _lines(log)
    assert [line["attempts"] for line in lines if line["kind"] == "error"] == [3]
    assert lines[-1]["retried"] == 1
    assert (lines[-1]["stop"], lines[-1]["retryable"], lines[-1]["rate_limited"]) == (
        "idle",
        True,
        True,
    )


def test_a_loop_blocked_past_an_idle_deadline_still_answers_every_call(tmp_path):
    """The reviewer's probe: call 2 lands and blocks the loop past call 1's idle deadline."""
    idle = 0.2
    numbers: list[int] = []

    def send(call: Call):
        numbers.append(len(numbers) + 1)
        me = numbers[-1]

        async def land() -> Reply:
            if me == 2:
                loop = asyncio.get_running_loop()
                landed = loop.create_future()

                def fire() -> None:
                    landed.set_result(None)
                    time.sleep(idle)

                loop.call_later(idle / 2, fire)
                await landed
                return reply([{"age": 40}] * 2)
            await asyncio.Event().wait()
            raise AssertionError(call)

        return land()

    log = _log(tmp_path)
    with (
        pytest.raises(IdleError),
        log,
        Flight(send, log, Limits(in_flight=2, idle=idle), spendable=lambda: None) as f,
    ):
        f.wave(PLAN)
    answers = _answered_lines(log)
    assert all(len(kinds) == 1 for kinds in answers.values()), answers
    assert _lines(log)[-1]["failed"] == sum(kinds == ["error"] for kinds in answers.values())


def test_an_interrupt_before_the_calls_beside_it_start_still_logs_them(tmp_path):
    sent: list[Call] = []

    def send(call: Call):
        sent.append(call)
        if len(sent) == 1:
            raise KeyboardInterrupt
        return Delayed()(call)

    log = _log(tmp_path)
    with (
        pytest.raises(KeyboardInterrupt),
        log,
        Flight(send, log, Limits(in_flight=3, idle=None), spendable=lambda: None) as flight,
    ):
        flight.wave(PLAN)
    lines = _lines(log)
    assert {line["call"]: line["error"] for line in lines if line["kind"] == "error"} == {
        1: "KeyboardInterrupt",
        2: "CancelledError",
        3: "CancelledError",
    }
    assert lines[-1]["status"] == "interrupted"
    assert lines[-1]["failed"] == 3


def test_a_further_interrupt_while_calls_land_still_logs_every_call(tmp_path):
    """Further interrupts raised while the first one's cancelled calls land, as a repeated Ctrl-C
    raises once the loop's own handler is gone, still leave every call an answer line."""

    def send(call: Call):
        number = _asked(call)

        async def land() -> Reply:
            if number == 1:
                await asyncio.sleep(0.05)
                raise KeyboardInterrupt
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise KeyboardInterrupt from None
            raise AssertionError(call)

        return land()

    log = _log(tmp_path)
    with (
        pytest.raises(KeyboardInterrupt),
        log,
        Flight(send, log, Limits(in_flight=3, idle=None), spendable=lambda: None) as flight,
    ):
        flight.wave(PLAN)
    assert _answered_lines(log) == {1: ["error"], 2: ["error"], 3: ["error"]}
    assert _lines(log)[-1]["status"] == "interrupted"


@pytest.mark.parametrize("idle", [0.5, None])
def test_a_sends_own_timeout_is_its_failure_and_not_the_idle_rule(tmp_path, caplog, idle):
    def send(call: Call):
        async def time_out() -> Reply:
            # As httpx raises when one HTTP attempt times out and the SDK gives up.
            raise TimeoutError(call.user)

        return time_out()

    log = _log(tmp_path)
    caplog.set_level(logging.INFO)
    with (
        pytest.raises(TimeoutError) as raised,
        log,
        Flight(send, log, Limits(in_flight=1, idle=idle), spendable=lambda: None) as f,
    ):
        f.wave(PLAN[:1])
    assert not isinstance(raised.value, IdleError)
    assert not [m for m in caplog.messages if m.startswith("stop rule=")]
    lines = _lines(log)
    assert [line["error"] for line in lines if line["kind"] == "error"] == ["TimeoutError"]
    assert lines[-1]["stop"] is None


def test_a_send_that_replies_within_the_idle_limit_is_answered(tmp_path):
    with (
        _log(tmp_path) as log,
        Flight(Delayed(), log, Limits(in_flight=3, idle=5.0), spendable=lambda: None) as flight,
    ):
        assert [batch.index for batch in flight.wave(PLAN)] == [1, 2, 3]


def test_each_send_sees_its_call_number(tmp_path):
    seen: list[int] = []

    def send(call: Call):
        async def land() -> Reply:
            await asyncio.sleep(0)
            seen.append(CALL_NUMBER.get())
            return reply([{"age": 40}] * _asked(call))

        return land()

    token = CALL_NUMBER.set(0)
    try:
        with (
            _log(tmp_path) as log,
            Flight(send, log, Limits(in_flight=3, idle=None), spendable=lambda: None) as flight,
        ):
            flight.wave(PLAN)
            assert CALL_NUMBER.get() == 0
    finally:
        CALL_NUMBER.reset(token)
    assert sorted(seen) == [1, 2, 3]
