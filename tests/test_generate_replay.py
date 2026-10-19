"""`bko.generate.replay.Replay`: a call log's calls answered in order, each checked."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import pytest

from bko.generate.calls import ask
from bko.generate.errors import GenerationError, ReplayError
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.prompt import render
from bko.generate.replay import Replay
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply


def call_for(guidance: Guidance, rows: int) -> Call:
    """The call `ask` sends for `rows` records of `guidance`'s cell."""
    return Call(SETTINGS, guidance.sex, guidance.label, *render(guidance, rows, SETTINGS.guidance))


def logged(tmp_path: Path, *script: Reply | GenerationError) -> Path:
    """A call log of one `ask` for 2 records of `GUIDANCE` per scripted outcome."""
    path = tmp_path / "paid.jsonl"
    with CallLog(path, SETTINGS) as log:
        for outcome in script:
            with contextlib.suppress(GenerationError):
                ask(GUIDANCE, 2, send=ScriptedSend(outcome), log=log)
    return path


def test_a_replayed_reply_equals_the_logged_reply(tmp_path):
    original = dataclasses.replace(reply([young(1), young(1)]), attempts=2)
    replay = Replay(logged(tmp_path, original))
    with CallLog(tmp_path / "again.jsonl", SETTINGS, replay=replay.digest) as log:
        batch = ask(GUIDANCE, 2, send=replay, log=log)
    assert batch.reply == original
    assert replay.remaining == 0


def test_replay_answers_one_request_in_the_logged_order(tmp_path):
    first = reply([young(1), young(1)], request_id="req_1")
    second = reply([young(1), young(1)], request_id="req_2")
    replay = Replay(logged(tmp_path, first, second))
    call = call_for(GUIDANCE, 2)
    assert [replay(call).request_id, replay(call).request_id] == ["req_1", "req_2"]


def test_a_logged_failure_is_raised_again(tmp_path):
    failure = GenerationError("the API could not be reached: X", attempts=3)
    replay = Replay(logged(tmp_path, failure))
    with pytest.raises(GenerationError, match="could not be reached") as raised:
        replay(call_for(GUIDANCE, 2))
    assert raised.value.attempts == 3


def test_a_call_that_differs_is_refused(tmp_path):
    replay = Replay(logged(tmp_path, reply([young(1), young(1)])))
    with pytest.raises(ReplayError, match="differs"):
        replay(call_for(GUIDANCE, 3))


def test_a_call_past_the_end_is_refused(tmp_path):
    replay = Replay(logged(tmp_path))
    with pytest.raises(ReplayError, match="no call left"):
        replay(call_for(GUIDANCE, 2))


def test_a_log_of_another_schema_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    lines = path.read_text().splitlines()
    run = json.loads(lines[0]) | {"schema": 1}
    path.write_text("\n".join([json.dumps(run), *lines[1:]]) + "\n")
    with pytest.raises(ReplayError, match="schema 2"):
        Replay(path)


def test_a_line_that_is_not_json_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path.write_text(path.read_text() + "{\n")
    with pytest.raises(ReplayError, match="not JSON"):
        Replay(path)


def test_a_request_with_no_answer_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    lines = path.read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["kind"] != "reply"]
    path.write_text("\n".join(kept) + "\n")
    with pytest.raises(ReplayError, match="no reply or error"):
        Replay(path)


def rewritten(path: Path, edit) -> Path:
    """Rewrite `path`'s JSON lines after `edit` changed the list in place."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(lines)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def first(lines: list[dict], kind: str) -> dict:
    """The first line of `kind`."""
    return next(line for line in lines if line["kind"] == kind)


def test_a_line_that_is_not_an_object_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path.write_text(path.read_text() + "[1]\n")
    with pytest.raises(ReplayError, match="not a JSON object"):
        Replay(path)


def test_an_empty_log_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ReplayError, match="not a schema 2 call log"):
        Replay(path)


def test_a_log_whose_first_line_is_not_its_run_line_is_refused(tmp_path):
    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), lambda lines: lines.reverse())
    with pytest.raises(ReplayError, match="not a schema 2 call log"):
        Replay(path)


@pytest.mark.parametrize("kind", ["request", "reply"])
@pytest.mark.parametrize("call", [None, "1", 1.0, True])
def test_a_line_without_a_whole_number_call_is_refused(tmp_path, kind, call):
    def edit(lines: list[dict]) -> None:
        line = first(lines, kind)
        if call is None:
            del line["call"]
        else:
            line["call"] = call

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="no whole-number call"):
        Replay(path)


@pytest.mark.parametrize("kind", ["request", "reply"])
def test_a_repeated_call_is_refused_not_collapsed(tmp_path, kind):
    def edit(lines: list[dict]) -> None:
        lines.insert(-1, dict(first(lines, kind)))

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="more than one"):
        Replay(path)


def test_calls_that_skip_an_index_are_refused(tmp_path):
    def edit(lines: list[dict]) -> None:
        for line in lines[1:-1]:
            line["call"] += 1

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="not calls 1 to 1"):
        Replay(path)


def test_an_answer_with_no_request_is_refused(tmp_path):
    def edit(lines: list[dict]) -> None:
        lines.insert(-1, first(lines, "reply") | {"call": 2})

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="no request line"):
        Replay(path)


@pytest.mark.parametrize(
    ("kind", "key"),
    [
        ("request", "params"),
        ("reply", "usage"),
        ("reply", "response"),
        ("reply", "attempts"),
        ("reply", "request_id"),
        ("error", "error"),
        ("error", "attempts"),
    ],
)
def test_a_line_missing_a_field_is_refused(tmp_path, kind, key):
    outcome = GenerationError("x") if kind == "error" else reply([young(1), young(1)])
    path = rewritten(logged(tmp_path, outcome), lambda lines: first(lines, kind).pop(key))
    with pytest.raises(ReplayError, match=f"call 1 of {path}"):
        Replay(path)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0},
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "server_tool_use": 0,
        },
        {
            "input_tokens": 1.5,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    ],
)
def test_a_reply_whose_usage_is_not_usage_is_refused(tmp_path, usage):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path = rewritten(path, lambda lines: first(lines, "reply").update(usage=usage))
    with pytest.raises(ReplayError, match="usage"):
        Replay(path)
