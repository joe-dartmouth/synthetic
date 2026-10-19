"""`bko.generate.calls`: `ask` makes one logged call, `generate` loops a quota in fixed chunks."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import NoReturn

import pytest

from bko.gaps.observe import Cell
from bko.generate import calls as calls_module
from bko.generate.calls import ask, generate
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import GenerationError
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Level, Reply, Usage
from bko.generate.prompt import render
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import (
    CELL,
    GUIDANCE,
    OTHER,
    OTHER_GUIDANCE,
    ScriptedSend,
    reply,
)


GUIDED: dict[Cell, Guidance] = {CELL: GUIDANCE, OTHER: OTHER_GUIDANCE}


def _log(tmp_path, settings=SETTINGS) -> CallLog:
    return CallLog(tmp_path / "calls.jsonl", settings, now=lambda: FIXED_NOW)


def _asked(call: Call) -> int:
    """The record count the call's user turn asks for."""
    return int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])


def test_ask_renders_at_the_log_settings_level_and_returns_candidates_and_spend(tmp_path):
    settings = dataclasses.replace(SETTINGS, guidance=Level.CELL)
    send = ScriptedSend(reply([{"age": 40}], usage=Usage(1_000_000, 0, 0, 0)))
    with _log(tmp_path, settings) as log:
        batch = ask(GUIDANCE, 2, send=send, log=log)
    system, user = render(GUIDANCE, 2, Level.CELL)
    assert send.calls == [Call(settings, GUIDANCE.sex, GUIDANCE.label, system, user)]
    assert batch.index == 1
    assert batch.cell == CELL
    assert batch.asked == 2
    assert batch.candidates == ({"age": 40}, Undecoded(Reason.SHORT))
    assert batch.undecoded == 1
    assert batch.usd == pytest.approx(4.00)


def test_generate_makes_ceil_rows_over_per_call_calls_per_cell_in_quota_order(tmp_path, caplog):
    send = ScriptedSend(*(reply([{}] * n) for n in (3, 20, 20, 5)))
    with caplog.at_level(logging.INFO), _log(tmp_path) as log:
        generation = generate({OTHER: 3, CELL: 45}, GUIDED, send=send, log=log)
    # 4 calls x 16000 output tokens x 20 USD per million output tokens.
    assert "plan calls=4 level=ontology max_tokens=16000 max_output_usd=1.28" in caplog.text
    assert [(call.sex, _asked(call)) for call in send.calls] == [(1, 3), (0, 20), (0, 20), (0, 5)]
    assert [batch.index for batch in generation.batches] == [1, 2, 3, 4]
    assert len(generation.candidates(CELL)) == 45
    assert len(generation.candidates(OTHER)) == 3
    assert generation.usd == pytest.approx(4 * 0.014)


@pytest.mark.parametrize(
    "quota",
    [
        {("<55", "female", 0): 2},
        {CELL: 0},
        {CELL: True},
        {CELL: 2.0},
        {CELL: 2, OTHER: -1},
    ],
    ids=["unknown-cell", "zero", "bool", "float", "negative-after-a-good-entry"],
)
def test_generate_refuses_a_bad_quota_before_any_call(tmp_path, quota):
    send = ScriptedSend()
    with _log(tmp_path) as log, pytest.raises(ValueError, match="quota"):
        generate(quota, GUIDED, send=send, log=log)
    assert send.calls == []


def test_a_failure_mid_run_raises_after_logging_and_keeps_earlier_lines(tmp_path):
    send = ScriptedSend(reply([{}] * 20), GenerationError("the API could not be reached"))
    with pytest.raises(GenerationError), _log(tmp_path) as log:
        generate({CELL: 25}, GUIDED, send=send, log=log)
    lines = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()
    ]
    kinds = [line["kind"] for line in lines]
    assert kinds == ["run", "request", "reply", "request", "error", "end"]
    assert lines[2]["decoded"] == 20
    assert lines[-1]["status"] == "failed"


def test_a_served_model_other_than_the_settings_model_warns(tmp_path, caplog):
    send = ScriptedSend(reply([{}], model="claude-other"))
    with caplog.at_level(logging.INFO), _log(tmp_path) as log:
        ask(GUIDANCE, 1, send=send, log=log)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert [record.getMessage() for record in warnings] == [
        f"served call=1 model=claude-other settings={SETTINGS.model}"
    ]


def test_a_short_call_warns_with_its_reason(tmp_path, caplog):
    send = ScriptedSend(reply([], stop_reason="refusal"))
    with caplog.at_level(logging.INFO), _log(tmp_path) as log:
        ask(GUIDANCE, 2, send=send, log=log)
    messages = [record.getMessage() for record in caplog.records]
    assert "short call=1 cell=<55/female/1 asked=2 undecoded=2 reason=refused" in messages
    assert (
        "generate call=1 cell=<55/female/1 level=ontology asked=2 decoded=0 undecoded=2 "
        "stop_reason=refusal input_tokens=1000 output_tokens=500 usd=0.0140 request_id=req_test"
    ) in messages


@pytest.mark.parametrize("call", ["ask", "generate"])
@pytest.mark.parametrize(
    "settings",
    [
        dataclasses.replace(SETTINGS, model="claude-other"),
        dataclasses.replace(SETTINGS, guidance=Level.CELL),
    ],
    ids=["model", "guidance"],
)
def test_settings_other_than_the_logs_are_refused_before_the_request_line(tmp_path, call, settings):
    send = ScriptedSend()

    def run(log: CallLog) -> None:
        if call == "ask":
            ask(GUIDANCE, 2, send=send, log=log, settings=settings)
        else:
            generate({CELL: 2}, GUIDED, send=send, log=log, settings=settings)

    with _log(tmp_path) as log, pytest.raises(ValueError, match="settings"):
        run(log)
    assert send.calls == []
    lines = (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["run", "end"]


def test_any_exception_from_send_writes_an_error_line_and_propagates(tmp_path):
    secret = "sk-ant-not-a-real-key"

    def send(call: Call) -> Reply:
        raise RuntimeError(secret)

    with pytest.raises(RuntimeError, match=secret), _log(tmp_path) as log:
        ask(GUIDANCE, 2, send=send, log=log)
    text = (tmp_path / "calls.jsonl").read_text("utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    assert [line["kind"] for line in lines] == ["run", "request", "error", "end"]
    assert lines[2]["error"] == "RuntimeError"
    assert lines[3]["failed"] == 1
    assert secret not in text


def test_a_send_line_precedes_each_call_so_sdk_retry_lines_sit_between_it_and_the_result(
    tmp_path, caplog
):
    seen: list[list[str]] = []

    def send(call: Call) -> Reply:
        seen.append([record.getMessage() for record in caplog.records])
        return reply([{}, {}])

    with caplog.at_level(logging.INFO), _log(tmp_path) as log:
        ask(GUIDANCE, 2, send=send, log=log)
    assert seen[0][-1] == "send call=1 cell=<55/female/1 asked=2"


@pytest.mark.parametrize("raised", [KeyboardInterrupt, ValueError])
def test_anything_raised_after_the_request_line_writes_the_error_line_and_propagates(
    tmp_path, caplog, monkeypatch, raised
):
    """A `KeyboardInterrupt` in `send`, or a failure reading a billed reply, leaves no request
    line unpaired."""

    def fail(*_: object) -> NoReturn:
        raise raised("after the reply")

    if raised is ValueError:
        monkeypatch.setattr(calls_module, "cost", fail)
        send = ScriptedSend(reply([{}, {}]))
    else:
        send = fail
    with caplog.at_level(logging.INFO), pytest.raises(raised), _log(tmp_path) as log:
        ask(GUIDANCE, 2, send=send, log=log)
    lines = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()
    ]
    assert [line["kind"] for line in lines] == ["run", "request", "error", "end"]
    assert lines[2]["error"] == raised.__name__
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert errors == [f"failed call=1 cell=<55/female/1 error={raised.__name__}"]
