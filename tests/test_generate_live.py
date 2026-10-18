"""One real call: the API accepts the request and answers in the schema's shape.

Paid, so it runs only under `--live`. It checks structure only, never values.
"""

from __future__ import annotations

import json
import logging

import anthropic
import pytest

from bko.datasets.uci import FRAME_COLUMNS
from bko.generate.api import send_with
from bko.generate.messages import SETTINGS, Call, Level
from bko.generate.prompt import render
from tests._generate_fixtures import GUIDANCE


pytestmark = pytest.mark.live

logger = logging.getLogger(__name__)


def test_one_real_call_returns_schema_shaped_records(tmp_path):
    call = Call(SETTINGS, 0, 1, *render(GUIDANCE, 2, Level.ONTOLOGY))
    reply = send_with(anthropic.Anthropic())(call)
    (tmp_path / "call.json").write_text(
        json.dumps({"params": call.params(), "raw": reply.raw}, indent=2), encoding="utf-8"
    )
    logger.info(
        "request_id=%s model=%s usage=%s call=%s",
        reply.request_id,
        reply.model,
        reply.usage,
        tmp_path / "call.json",
    )
    assert reply.stop_reason == "end_turn"
    assert reply.text is not None
    records = json.loads(reply.text)["records"]
    assert isinstance(records, list)
    assert len(records) >= 2
    for record in records:
        assert set(record) == set(FRAME_COLUMNS)
        assert record["sex"] == 0
        assert record["target"] == 1
    assert reply.request_id is not None
    assert reply.request_id.startswith("req_")
    assert reply.usage.output_tokens <= SETTINGS.max_tokens
