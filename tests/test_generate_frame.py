"""Generation through the SDK adapter follows the call's frame: what is sent is what is logged."""

from __future__ import annotations

from typing import Any

import anthropic
import pytest

from bko.generate.api import send_with
from bko.generate.messages import SETTINGS, Call, record_schema
from tests._frame_fixtures import TOY
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
