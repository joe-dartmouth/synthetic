"""Generation through the SDK adapter follows the call's frame: what is sent is what is logged."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

import anthropic
import pytest

from bko.baselines.request import Request
from bko.evaluate.guided import SEQUENTIAL, GuidedArm, Knowledge, spendable
from bko.gaps.definitions import Definition
from bko.generate.api import send_with
from bko.generate.flight import Flight
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, record_schema
from bko.generate.prompt import system
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN
from tests._generate_fixtures import message, records_text


def fake_client(monkeypatch: pytest.MonkeyPatch, records: list[object]) -> tuple[Any, list]:
    """A client whose `messages.create` records each request and answers with `records`."""
    client = anthropic.Anthropic(api_key="not-a-credential", max_retries=0)
    sent: list[dict[str, object]] = []

    def create(**kwargs: object) -> anthropic.types.Message:
        sent.append(kwargs)
        return message(records_text(records))

    monkeypatch.setattr(client.messages, "create", create)
    return client, sent


def test_send_with_sends_the_toy_calls_params(monkeypatch):
    client, sent = fake_client(monkeypatch, [])
    call = Call(SETTINGS, 0, 1, "the system", "the user", frame=TOY)
    send_with(client)(call)
    assert sent == [call.params()]
    body: Any = sent[0]
    assert body["output_config"]["format"]["schema"] == record_schema(0, 1, frame=TOY)


TOY_ROW = {"age": 50, "sex": 0, "level": 1.5, "target": 1}
"""A toy record of `SHORT[0]`, inside every toy training range."""


def test_a_guided_arm_fills_a_toy_request_through_the_sdk_adapter(monkeypatch, tmp_path):
    client, sent = fake_client(monkeypatch, [TOY_ROW, TOY_ROW | {"level": 1.25}])
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS)
    flight = Flight(send_with(client), log, SEQUENTIAL, spendable=partial(spendable, log, 50.0))
    arm = GuidedArm(
        "bko", Definition.SUBGROUP, Knowledge("toy", {}, (), ()), log=log, budget=3, flight=flight
    )
    with log:
        baseline = arm(TOY_TRAIN, Request(quota={SHORT[0]: 2}, frame=TOY, seed=0))
    assert len(sent) == 1
    body: Any = sent[0]
    assert body["output_config"]["format"]["schema"] == record_schema(0, 1, frame=TOY)
    assert body["system"] == system(TOY)
    user = body["messages"][0]["content"]
    assert user.startswith(
        "Write exactly 2 records of patients in this group: sex 0 (female), target 1 (toy "
        "condition present), age 45 to 69 years."
    )
    assert "level 0.5 to 4.5 mg/dL" in user
    logged = [
        line for line in log.path.read_text(encoding="utf-8").splitlines() if '"params"' in line
    ]
    assert len(logged) == 1
    assert json.loads(logged[0])["params"] == json.loads(json.dumps(body))
    assert baseline.added.columns.tolist() == ["age", "sex", "level", "target"]
    assert baseline.added["level"].tolist() == [1.5, 1.25]
