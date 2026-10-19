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
