"""`bko.generate.reasoning`: a local thinking reply's reasoning length and its per-call line."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from bko.generate.decode import decode, undecoded_in
from bko.generate.local import local_send, reply_of
from bko.generate.messages import SETTINGS
from bko.generate.reasoning import reasoning_chars, reasoning_text
from bko.generate.weights import LOCAL_MODELS
from tests.test_generate_local import (
    BASE,
    CALL,
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


@pytest.mark.parametrize(
    "answer",
    [(500, {"error": "busy"}), (200, {"n": 3}), (200, {"tokens": 3}), (200, b"not json")],
    ids=["status", "no-tokens", "tokens-not-a-list", "not-json"],
)
def test_a_tokenize_failure_logs_unknown_and_keeps_the_reply(answer, caplog):
    caplog.set_level("INFO")
    body = completion([RECORD], reasoning_content="think")
    with _thinking_send(body, [answer]) as fake:
        reply = asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    records = _reasoning_records(caplog)
    assert [level for level, _ in records] == ["INFO"]
    assert " reasoning_chars=5 reasoning_tokens=unknown " in records[0][1]
    candidates = decode(reply, 1)
    assert candidates[: len(candidates) - undecoded_in(candidates)] == (RECORD,)


def test_a_base_call_logs_no_reasoning_line(caplog):
    caplog.set_level("INFO")
    props = {**PROPS, "model_alias": BASE, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    base_call = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=BASE))
    body = completion([RECORD], reasoning_content="stray")
    with Server([(200, body)], props=props) as fake:
        asyncio.run(local_send(BASE, url=fake.url)(base_call))
    assert fake.tokenized == []
    assert _reasoning_records(caplog) == []
