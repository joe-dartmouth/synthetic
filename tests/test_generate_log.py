"""`bko.generate.log.CallLog`: an append-only JSONL record of every call, kept through failure."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import Any, TextIO

import pytest

from bko.generate.calls import ask
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import GenerationError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply, Usage
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import CELL, GUIDANCE, ScriptedSend, reply


def _now():
    return FIXED_NOW


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _call() -> Call:
    return Call(SETTINGS, GUIDANCE.sex, GUIDANCE.label, "system", "user")


def test_an_unpriced_model_is_refused_before_the_file_exists(tmp_path):
    path = tmp_path / "calls.jsonl"
    settings = dataclasses.replace(SETTINGS, model="claude-unpriced")
    with pytest.raises(GenerationError, match="claude-unpriced"):
        CallLog(path, settings, now=_now)
    assert not path.exists()


def test_the_header_holds_schema_settings_price_and_sdk_version(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now):
        pass
    header = _lines(path)[0]
    assert header == {
        "kind": "run",
        "schema": 2,
        "replay": None,
        "started": FIXED_NOW.isoformat(),
        "settings": {
            "model": "claude-opus-5-5",
            "effort": "medium",
            "max_tokens": 16000,
            "per_call": 20,
            "guidance": "ontology",
        },
        "price": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.2},
        "anthropic": importlib.metadata.version("anthropic"),
    }


def test_a_replay_log_names_its_source(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, replay="ab" * 32, now=_now):
        pass
    assert _lines(path)[0]["replay"] == "ab" * 32


def test_a_request_line_is_flushed_before_the_call(tmp_path):
    path = tmp_path / "calls.jsonl"
    seen: list[list[dict[str, Any]]] = []

    def send(call: Call) -> Reply:
        seen.append(_lines(path))
        return reply([{}, {}])

    with CallLog(path, SETTINGS, now=_now) as log:
        ask(GUIDANCE, 2, send=send, log=log)
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
    with CallLog(path, SETTINGS, now=_now) as log:
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
    with pytest.raises(GenerationError), CallLog(path, SETTINGS, now=_now) as log:
        ask(GUIDANCE, 2, send=send, log=log)
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
    with CallLog(path, SETTINGS, now=_now) as log:
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
    with pytest.raises(raised), CallLog(path, SETTINGS, now=_now):
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
        ask(GUIDANCE, 3, send=send, log=log)
        ask(GUIDANCE, 4, send=send, log=log)

    with pytest.raises(GenerationError), CallLog(path, SETTINGS, now=_now) as log:
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
        "calls": 2,
        "failed": 1,
        "retried": 0,
        "asked": 7,
        "decoded": 2,
        "undecoded": 1,
        "input_tokens": 1000,
        "output_tokens": 500,
        "usd": pytest.approx(0.014),
    }


def test_the_end_totals_count_the_calls_retried(tmp_path):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(
        dataclasses.replace(reply([{}, {}]), attempts=3),
        dataclasses.replace(reply([{}, {}]), attempts=1),
    )
    with CallLog(path, SETTINGS, now=_now) as log:
        ask(GUIDANCE, 2, send=send, log=log)
        ask(GUIDANCE, 2, send=send, log=log)
    assert _lines(path)[-1]["retried"] == 1


def test_a_second_log_on_the_same_path_is_refused_and_leaves_the_first_intact(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallLog(path, SETTINGS, now=_now):
        pass
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        CallLog(path, SETTINGS, now=_now)
    assert path.read_bytes() == before


def test_the_run_line_logs_status_totals_and_the_log_path(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    send = ScriptedSend(reply([{}, {}]))
    with caplog.at_level(logging.INFO), CallLog(path, SETTINGS, now=_now) as log:
        ask(GUIDANCE, 2, send=send, log=log)
    assert (
        "run status=complete calls=1 failed=0 retried=0 asked=2 decoded=2 undecoded=0 "
        f"input_tokens=1000 output_tokens=500 usd=0.0140 replay=none log={path}"
    ) in caplog.text


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
        CallLog(path, SETTINGS, now=_now)
    assert len(opened) == 1
    assert opened[0].closed


def test_the_log_exposes_the_settings_it_priced(tmp_path):
    with CallLog(tmp_path / "calls.jsonl", SETTINGS, now=_now) as log:
        assert log.settings is SETTINGS


def test_the_log_is_readable_by_its_owner_only(tmp_path):
    path = tmp_path / "calls.jsonl"
    previous = os.umask(0o022)
    try:
        with CallLog(path, SETTINGS, now=_now):
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
        CallLog(path, SETTINGS, now=_now)
    assert not path.exists()


def test_opening_the_log_logs_its_path_settings_and_replay(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with caplog.at_level(logging.INFO), CallLog(path, SETTINGS, replay="ab12", now=_now):
        pass
    assert (
        f"log path={path} schema=2 model={SETTINGS.model} effort={SETTINGS.effort} "
        f"level={SETTINGS.guidance} per_call={SETTINGS.per_call} "
        f"max_tokens={SETTINGS.max_tokens} replay=ab12"
    ) in caplog.messages


def test_a_replays_end_line_and_totals_name_the_replayed_log(tmp_path, caplog):
    path = tmp_path / "calls.jsonl"
    with caplog.at_level(logging.INFO), CallLog(path, SETTINGS, replay="ab12", now=_now):
        pass
    assert _lines(path)[-1]["replay"] == "ab12"
    assert any(m.startswith("run status=complete") and "replay=ab12" in m for m in caplog.messages)
