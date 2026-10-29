"""`bko.generate.log.CallLog`: an append-only JSONL record of every call, kept through failure."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import Any, TextIO

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.generate.calls import ask
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import GenerationError
from bko.generate.log import Answered, CallLog
from bko.generate.messages import SETTINGS, Call, Reply, Usage
from bko.generate.spend import cost
from tests._datasets_fixtures import FIXED_NOW
from tests._fusion_fixtures import young
from tests._generate_fixtures import CELL, GUIDANCE, ScriptedSend, reply


def _now():
    return FIXED_NOW


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _call() -> Call:
    return Call(SETTINGS, GUIDANCE.sex, GUIDANCE.label, "system", "user", frame=UCI_FRAME)


def test_an_unpriced_model_is_refused_before_the_file_exists(tmp_path):
    path = tmp_path / "calls.jsonl"
    settings = dataclasses.replace(SETTINGS, model="claude-unpriced")
    with pytest.raises(GenerationError, match="claude-unpriced"):
        CallLog(path, settings, now=_now, frame="uci")
    assert not path.exists()


def test_the_header_holds_schema_settings_price_and_sdk_version(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now, frame="uci"):
        pass
    header = _lines(path)[0]
    assert header == {
        "kind": "run",
        "schema": 2,
        "replay": None,
        "resume": None,
        "rehearsal": False,
        "in_flight": 1,
        "frame": "uci",
        "started": FIXED_NOW.isoformat(),
        "settings": {
            "model": "claude-haiku-5-5",
            "effort": "medium",
            "max_tokens": 16000,
            "per_call": 20,
            "guidance": "summary",
        },
        "price": {"input": 0.1, "output": 0.5, "cache_write": 0.125, "cache_read": 0.01},
        "anthropic": importlib.metadata.version("anthropic"),
    }


def test_a_replay_log_names_its_source(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, answered=Answered("replay", "ab" * 32, 0), now=_now, frame="uci"):
        pass
    assert _lines(path)[0]["replay"] == "ab" * 32


def test_a_request_line_is_flushed_before_the_call(tmp_path):
    path = tmp_path / "calls.jsonl"
    seen: list[list[dict[str, Any]]] = []

    def send(call: Call) -> Reply:
        seen.append(_lines(path))
        return reply([{}, {}])

    with CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    assert [line["kind"] for line in seen[0]] == ["run", "request"]
    request = seen[0][1]
    assert request["call"] == 1
    assert request["cell"] == "<55/female/1"
    assert request["asked"] == 2
    assert request["at"] == FIXED_NOW.isoformat()
    assert request["params"]["model"] == SETTINGS.model
    assert request["params"]["output_config"]["effort"] == SETTINGS.effort


def test_a_reply_line_holds_request_id_model_stop_reason_counts_usage_usd_and_the_raw_response(
    tmp_path,
):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        index = log.request(CELL, 3, _call())
        answer = dataclasses.replace(
            reply([{}, {}], usage=Usage(1000, 500, 100, 2000), request_id="req_abc"), attempts=3
        )
        log.reply(index, answer, ({}, {}, Undecoded(Reason.SHORT)), 0.0149)
    line = _lines(path)[2]
    assert line == {
        "kind": "reply",
        "call": 1,
        "at": FIXED_NOW.isoformat(),
        "request_id": "req_abc",
        "attempts": 3,
        "model": SETTINGS.model,
        "stop_reason": "end_turn",
        "decoded": 2,
        "undecoded": 1,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 2000,
        },
        "usd": 0.0149,
        "response": {
            "id": "msg_test",
            "model": SETTINGS.model,
            "content": [{"type": "text", "text": '{"records": [{}, {}]}'}],
        },
    }


def test_a_failed_call_writes_an_error_line_and_a_failed_footer(tmp_path):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(
        GenerationError("the API could not be reached: APIConnectionError", attempts=2)
    )
    with pytest.raises(GenerationError), CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    lines = _lines(path)
    assert [line["kind"] for line in lines] == ["run", "request", "error", "end"]
    assert lines[2] == {
        "kind": "error",
        "call": 1,
        "at": FIXED_NOW.isoformat(),
        "error": "the API could not be reached: APIConnectionError",
        "attempts": 2,
    }
    assert lines[3]["status"] == "failed"
    assert lines[3]["failed"] == 1


def test_an_interrupt_writes_an_error_line_with_no_attempts(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        index = log.request(CELL, 2, _call())
        log.failure(index, KeyboardInterrupt())
    line = _lines(path)[2]
    assert line["kind"] == "error"
    assert line["attempts"] is None


@pytest.mark.parametrize(
    ("raised", "status"),
    [(KeyboardInterrupt, "interrupted"), (SystemExit, "failed"), (RuntimeError, "failed")],
)
def test_a_keyboard_interrupt_writes_interrupted_and_any_other_exception_failed(
    tmp_path, raised, status
):
    path = tmp_path / "calls.jsonl"
    with pytest.raises(raised), CallLog(path, SETTINGS, now=_now, frame="uci"):
        raise raised
    end = _lines(path)[-1]
    assert end["kind"] == "end"
    assert end["status"] == status
    assert end["ended"] == FIXED_NOW.isoformat()


def test_every_line_is_json_and_totals_survive_a_failure(tmp_path):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(
        reply([{}, {}], usage=Usage(1000, 500, 0, 0)),
        GenerationError("the API refused the request"),
    )

    def two_calls(log: CallLog) -> None:
        ask(GUIDANCE, 3, send=send, log=log, frame=UCI_FRAME)
        ask(GUIDANCE, 4, send=send, log=log, frame=UCI_FRAME)

    with pytest.raises(GenerationError), CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        two_calls(log)
    lines = _lines(path)
    kinds = [line["kind"] for line in lines]
    assert kinds == ["run", "request", "reply", "request", "error", "end"]
    send.assert_consumed()
    end = lines[-1]
    del end["ended"]
    assert end == {
        "kind": "end",
        "status": "failed",
        "replay": None,
        "resume": None,
        "rehearsal": False,
        "stop": None,
        "retryable": False,
        "rate_limited": False,
        "calls": 2,
        "failed": 1,
        "retried": 0,
        "asked": 7,
        "decoded": 2,
        "undecoded": 1,
        "input_tokens": 1000,
        "output_tokens": 500,
        "usd": pytest.approx(0.00035),
        "live_usd": pytest.approx(0.00035),
    }


def test_the_end_totals_count_the_calls_retried(tmp_path):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(
        dataclasses.replace(reply([{}, {}]), attempts=3),
        dataclasses.replace(reply([{}, {}]), attempts=1),
    )
    with CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    assert _lines(path)[-1]["retried"] == 1


def test_a_second_log_on_the_same_path_is_refused_and_leaves_the_first_intact(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now, frame="uci"):
        pass
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        CallLog(path, SETTINGS, now=_now, frame="uci")
    assert path.read_bytes() == before


def test_the_run_line_logs_status_totals_and_the_log_path(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(reply([{}, {}]))
    with caplog.at_level(logging.INFO), CallLog(path, SETTINGS, now=_now, frame="uci") as log:
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    assert (
        "run status=complete run=none level=summary in_flight=1 calls=1 failed=0 retried=0 "
        "asked=2 decoded=2 undecoded=0 "
        f"input_tokens=1000 output_tokens=500 usd=0.000350 live_usd=0.000350 replay=none "
        f"resume=none rehearsal=false log={path}"
    ) in caplog.text


def test_the_run_status_line_names_run_level_and_in_flight(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with (
        caplog.at_level(logging.INFO),
        CallLog(path, SETTINGS, run="g3-cell", in_flight=4, frame="uci"),
    ):
        pass
    assert any(
        m.startswith(
            f"run status=complete run=g3-cell level={SETTINGS.guidance} in_flight=4 calls=0 "
        )
        for m in caplog.messages
    )


def test_a_failed_header_write_closes_the_file(tmp_path, monkeypatch):
    opened: list[TextIO] = []
    real_fdopen = os.fdopen

    def spy(*args: Any, **kwargs: Any) -> Any:
        handle = real_fdopen(*args, **kwargs)
        opened.append(handle)
        return handle

    def refuse(self: CallLog, line: dict[str, object]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fdopen", spy)
    monkeypatch.setattr(CallLog, "_write", refuse)
    path = tmp_path / "calls.jsonl"
    with pytest.raises(OSError, match="disk full"):
        CallLog(path, SETTINGS, now=_now, frame="uci")
    assert len(opened) == 1
    assert opened[0].closed
    assert not path.exists()


def test_a_log_whose_file_cannot_be_opened_is_removed(tmp_path, monkeypatch):
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no descriptor")

    monkeypatch.setattr(os, "fdopen", refuse)
    path = tmp_path / "calls.jsonl"
    with pytest.raises(OSError, match="no descriptor"):
        CallLog(path, SETTINGS, now=_now, frame="uci")
    assert not path.exists()


def test_the_log_exposes_the_settings_it_priced(tmp_path):
    with CallLog(tmp_path / "calls.jsonl", SETTINGS, now=_now, frame="uci") as log:
        assert log.settings is SETTINGS


def test_the_log_is_readable_by_its_owner_only(tmp_path):
    path = tmp_path / "calls.jsonl"
    previous = os.umask(0o022)
    try:
        with CallLog(path, SETTINGS, now=_now, frame="uci"):
            pass
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_missing_sdk_version_leaves_no_file(tmp_path, monkeypatch):
    def absent(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    path = tmp_path / "calls.jsonl"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        CallLog(path, SETTINGS, now=_now, frame="uci")
    assert not path.exists()


def test_opening_the_log_logs_its_path_settings_and_replay(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with (
        caplog.at_level(logging.INFO),
        CallLog(path, SETTINGS, answered=Answered("replay", "ab12", 0), now=_now, frame="uci"),
    ):
        pass
    assert (
        f"log path={path} schema=2 model={SETTINGS.model} effort={SETTINGS.effort} "
        f"level={SETTINGS.guidance} per_call={SETTINGS.per_call} "
        f"max_tokens={SETTINGS.max_tokens} replay=ab12 resume=none rehearsal=false in_flight=1"
    ) in caplog.messages


def test_the_log_records_the_calls_in_flight(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with caplog.at_level(logging.INFO), CallLog(path, SETTINGS, in_flight=8, now=_now, frame="uci"):
        pass
    assert _lines(path)[0]["in_flight"] == 8
    assert any(m.startswith("log path=") and m.endswith(" in_flight=8") for m in caplog.messages)


def test_a_replays_end_line_and_totals_name_the_replayed_log(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with (
        caplog.at_level(logging.INFO),
        CallLog(path, SETTINGS, answered=Answered("replay", "ab12", 0), now=_now, frame="uci"),
    ):
        pass
    assert _lines(path)[-1]["replay"] == "ab12"
    assert any(m.startswith("run status=complete") and "replay=ab12" in m for m in caplog.messages)


def test_a_rehearsals_run_and_end_lines_and_log_lines_say_so(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with (
        caplog.at_level(logging.INFO),
        CallLog(path, SETTINGS, rehearsal=True, now=_now, frame="uci"),
    ):
        pass
    lines = _lines(path)
    assert lines[0]["rehearsal"] is True
    assert lines[-1]["rehearsal"] is True
    assert sum("rehearsal=true" in message for message in caplog.messages) == 2


class StoppedError(Exception):
    def __init__(self, rule: object) -> None:
        super().__init__("stopped")
        self.stop_rule = rule


@pytest.mark.parametrize(
    ("raised", "stop"),
    [
        (StoppedError("idle"), "idle"),
        (StoppedError("projection"), "projection"),
        (StoppedError("ceiling"), "ceiling"),
        (StoppedError("other"), None),
        (ValueError("x"), None),
    ],
)
def test_the_end_line_names_the_stop_rule_that_ended_the_run(tmp_path, raised, stop):
    path = tmp_path / "calls.jsonl"
    with pytest.raises(type(raised)), CallLog(path, SETTINGS, now=_now, frame="uci"):
        raise raised
    assert _lines(path)[-1]["stop"] == stop


def test_a_complete_runs_end_line_names_no_stop(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now, frame="uci"):
        pass
    assert _lines(path)[-1]["stop"] is None


def test_live_usd_counts_only_calls_past_the_answered_ones(tmp_path):
    path = tmp_path / "log.jsonl"
    with CallLog(path, SETTINGS, answered=Answered("resume", "ab" * 32, 1), frame="uci") as log:
        for _ in range(2):
            ask(
                GUIDANCE,
                2,
                send=ScriptedSend(reply([young(1), young(1)])),
                log=log,
                frame=UCI_FRAME,
            )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    one = cost(reply([young(1), young(1)]).usage, SETTINGS.model)
    assert lines[0]["resume"] == {"sha256": "ab" * 32, "calls": 1}
    assert lines[0]["replay"] is None
    assert lines[-1]["usd"] == pytest.approx(2 * one)
    assert lines[-1]["live_usd"] == pytest.approx(one)


@pytest.mark.parametrize(
    ("outcome", "limited"),
    [
        (reply([{}, {}]), False),
        (dataclasses.replace(reply([{}, {}]), rate_limited=True), True),
        (GenerationError("the API refused the request", rate_limited=True), True),
        (GenerationError("the API refused the request"), False),
    ],
    ids=["reply", "rate-limited-reply", "rate-limited-failure", "failure"],
)
def test_the_end_line_marks_rate_limited_segments(tmp_path, outcome, limited):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(reply([{}, {}]), outcome)
    with (
        contextlib.suppress(GenerationError),
        CallLog(path, SETTINGS, now=_now, frame="uci") as log,
    ):
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
        ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    send.assert_consumed()
    assert _lines(path)[-1]["rate_limited"] is limited
