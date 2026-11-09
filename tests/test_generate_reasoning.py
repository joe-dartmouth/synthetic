"""`bko.generate.reasoning`: a local thinking reply's reasoning length and its per-call line."""

from __future__ import annotations

import asyncio
import dataclasses
import http.client
import json
import threading
import time

import pytest

import bko.generate.reasoning as reasoning_module
from bko.datasets.uci import UCI_FRAME
from bko.generate.calls import ask
from bko.generate.chat import reply_of
from bko.generate.decode import decode, undecoded_in
from bko.generate.local import local_send
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS
from bko.generate.reasoning import (
    FIELD_CHARS,
    field,
    log_reasoning,
    reasoning_chars,
    reasoning_text,
)
from bko.generate.replay import Replay
from bko.generate.weights import LOCAL_MODELS
from tests._generate_fixtures import (
    BASE,
    CALL,
    GUIDANCE,
    HANG_UP,
    PREFLIGHT,
    PROPS,
    RECORD,
    THINKING,
    THINKING_CALL,
    THINKING_PROPS,
    Server,
    completion,
)


@pytest.mark.parametrize("thoughts", [7, ["a"], {"a": 1}, True])
def test_reasoning_that_is_not_text_counts_zero(thoughts):
    reply = reply_of(completion([RECORD], reasoning_content=thoughts))
    assert (reasoning_text(reply), reasoning_chars(reply)) == (None, 0)


def test_reasoning_text_counts_its_characters():
    reply = reply_of(completion([RECORD], reasoning_content="t" * 50))
    assert (reasoning_text(reply), reasoning_chars(reply)) == ("t" * 50, 50)


def _thinking_send(body: dict[str, object], tokenize: list[tuple[int, object]]) -> Server:
    return Server([PREFLIGHT, (200, body)], props=THINKING_PROPS, tokenize=tokenize)


def _reasoning_records(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if r.name == "bko.generate.reasoning" and r.getMessage().startswith("reasoning ")
    ]


def test_a_thinking_call_logs_its_reasoning_share(caplog):
    caplog.set_level("INFO")
    body = completion([RECORD], reasoning_content="t" * 50)
    with _thinking_send(body, [(200, {"tokens": [0] * 12})]) as fake:
        reply = asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    assert fake.tokenized == [{"content": "t" * 50, "add_special": False}]
    content = len(json.dumps({"records": [RECORD]}))
    line = "reasoning request_id=chatcmpl-1 stop_reason=end_turn reasoning_chars=50"
    line += f" reasoning_tokens=12 content_chars={content} output_tokens=120"
    assert _reasoning_records(caplog) == [("INFO", line)]
    assert reply.text == json.dumps({"records": [RECORD]})


def test_a_reply_cut_while_reasoning_warns_with_its_reason(caplog):
    caplog.set_level("INFO")
    body = completion([], finish="length", reasoning_content="t" * 50, content="")
    with _thinking_send(body, [(200, {"tokens": [0] * 12})]) as fake:
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    records = _reasoning_records(caplog)
    assert [level for level, _ in records] == ["WARNING"]
    assert records[0][1].endswith(
        "stop_reason=max_tokens reasoning_chars=50 reasoning_tokens=12 content_chars=0 "
        "output_tokens=120 reason=truncated_while_reasoning"
    )


DEEP: bytes = b"[" * 1_000_000 + b"]" * 1_000_000
"""A JSON body nested past any parser's recursion limit."""


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        ((500, {"tokens": [0] * 3}), "status=500"),
        ((200, {"n": 3}), "KeyError"),
        ((200, {"tokens": 3}), "int"),
        ((200, b"not json"), "JSONDecodeError"),
        ((200, [1, 2]), "TypeError"),
        ((200, 3), "TypeError"),
        ((200, DEEP), "RecursionError"),
        (HANG_UP, "RemoteDisconnected"),
    ],
    ids=[
        "status",
        "no-tokens",
        "tokens-not-a-list",
        "not-json",
        "list",
        "scalar",
        "deep",
        "hang-up",
    ],
)
def test_a_tokenize_failure_logs_unknown_warns_once_and_keeps_the_reply(answer, why, caplog):
    caplog.set_level("INFO")
    body = completion([RECORD], reasoning_content="think")
    answers: list[tuple[int, object]] = [PREFLIGHT, (200, body), (200, body)]
    with Server(answers, props=THINKING_PROPS, tokenize=[answer, answer]) as fake:
        send = local_send(THINKING, url=fake.url)
        replies = [asyncio.run(send(THINKING_CALL)) for _ in range(2)]
    records = _reasoning_records(caplog)
    assert [level for level, _ in records] == ["INFO", "INFO"]
    assert all(" reasoning_chars=5 reasoning_tokens=unknown " in line for _, line in records)
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("tokenize ")]
    assert warnings == [f"tokenize failed: {why}; this call logs reasoning_tokens=unknown"]
    for reply in replies:
        candidates = decode(reply, 1)
        assert candidates[: len(candidates) - undecoded_in(candidates)] == (RECORD,)


def test_a_slow_tokenize_gives_up_at_its_own_timeout(monkeypatch, caplog):
    """`/tokenize` waits `TOKENIZE_SECONDS`, not the call's `TIMEOUT_SECONDS`."""
    caplog.set_level("INFO")
    monkeypatch.setattr(reasoning_module, "TOKENIZE_SECONDS", 0.3)
    body = completion([RECORD], reasoning_content="think")
    with Server([PREFLIGHT, (200, body)], props=THINKING_PROPS, tokenize_delay=3.0) as fake:
        started = time.monotonic()
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
        elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert " reasoning_tokens=unknown " in _reasoning_records(caplog)[0][1]
    assert "tokenize failed: TimeoutError; this call logs reasoning_tokens=unknown" in (
        caplog.messages
    )


@pytest.mark.parametrize(
    ("finish", "thoughts", "content", "level"),
    [
        ("length", "t" * 50, "", "WARNING"),
        ("stop", "t" * 50, "", "INFO"),
        ("length", "t" * 50, None, "INFO"),
        ("length", None, "", "INFO"),
    ],
    ids=["cut-while-reasoning", "ended", "has-content", "no-reasoning"],
)
def test_only_a_reply_cut_while_reasoning_warns(finish, thoughts, content, level, caplog):
    caplog.set_level("INFO")
    reply = reply_of(
        completion([RECORD], finish=finish, reasoning_content=thoughts, content=content)
    )
    with Server([]) as fake:
        connection = http.client.HTTPConnection("127.0.0.1", fake.port, timeout=5.0)
        try:
            log_reasoning(reply, connection, threading.Event())
        finally:
            connection.close()
    records = _reasoning_records(caplog)
    assert [found for found, _ in records] == [level]
    assert records[0][1].endswith(" reason=truncated_while_reasoning") == (level == "WARNING")


def test_a_thinking_reply_without_reasoning_counts_zero_tokens_and_tokenizes_nothing(caplog):
    caplog.set_level("INFO")
    with _thinking_send(completion([RECORD]), []) as fake:
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    assert fake.tokenized == []
    records = _reasoning_records(caplog)
    assert len(records) == 1
    assert " reasoning_chars=0 reasoning_tokens=0 " in records[0][1]


def test_a_crafted_request_id_or_stop_reason_stays_one_bare_field(caplog):
    caplog.set_level("INFO")
    body = completion([RECORD], finish="x y", reasoning_content="think")
    body["id"] = "a b\nreasoning request_id=forged\x07" + "z" * 200
    with _thinking_send(body, [(200, {"tokens": [0] * 2})]) as fake:
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    ((_, line),) = _reasoning_records(caplog)
    escaped = "a\\u0020b\\u000areasoning\\u0020request_id=forged\\u0007"
    request_id = escaped + "z" * (FIELD_CHARS - 3 - len(escaped)) + "..."
    assert line.startswith(f"reasoning request_id={request_id} stop_reason=x\\u0020y ")
    assert "\n" not in line
    assert (field("chatcmpl-1"), field(None), field("end_turn")) == (
        "chatcmpl-1",
        "None",
        "end_turn",
    )


def test_a_replayed_thinking_call_logs_no_reasoning_line(tmp_path, caplog):
    caplog.set_level("INFO")
    settings = dataclasses.replace(SETTINGS, model=THINKING)
    body = completion([RECORD, RECORD], reasoning_content="think")
    path = tmp_path / "calls.jsonl"
    with CallLog(path, settings, frame="uci") as log:
        ask(GUIDANCE, 2, send=lambda call: reply_of(body), log=log, frame=UCI_FRAME)
    caplog.clear()
    with CallLog(tmp_path / "again.jsonl", settings, frame="uci") as again:
        ask(GUIDANCE, 2, send=Replay(path), log=again, frame=UCI_FRAME)
    assert (again.decoded, again.undecoded) == (2, 0)
    assert _reasoning_records(caplog) == []


def test_a_base_call_logs_no_reasoning_line(caplog):
    caplog.set_level("INFO")
    props = {**PROPS, "model_alias": BASE, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    base_call = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=BASE))
    body = completion([RECORD], reasoning_content="stray")
    with Server([(200, body)], props=props) as fake:
        asyncio.run(local_send(BASE, url=fake.url)(base_call))
    assert fake.tokenized == []
    assert _reasoning_records(caplog) == []
