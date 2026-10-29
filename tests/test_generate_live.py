"""One real call: the API accepts the request and answers in the schema's shape.

Paid, so it runs only under `--live`. It checks structure only (keys, types and codes), never
plausibility.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from bko.datasets.uci import CODEBOOK, FRACTIONAL, FRAME_COLUMNS, UCI_FRAME
from bko.generate.api import default_send
from bko.generate.messages import SETTINGS, Call
from bko.generate.prompt import render
from tests._generate_fixtures import GUIDANCE


pytestmark = pytest.mark.live

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model", [SETTINGS.model, "claude-sonnet-5-5"])
def test_one_real_call_returns_schema_shaped_records(tmp_path, model):
    settings = dataclasses.replace(SETTINGS, model=model)
    call = Call(
        settings, 0, 1, *render(GUIDANCE, 2, settings.guidance, frame=UCI_FRAME), frame=UCI_FRAME
    )
    reply = default_send()(call)
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
        for name in FRAME_COLUMNS:
            value = record[name]
            assert not isinstance(value, bool), name
            assert isinstance(value, (int, float) if name in FRACTIONAL else int), name
        for name, codes in CODEBOOK.items():
            assert record[name] in codes, name
    assert reply.attempts is not None
    assert reply.attempts >= 1
    assert reply.request_id is not None
    assert reply.request_id.startswith("req_")
    assert reply.model == model
    assert reply.usage.output_tokens <= SETTINGS.max_tokens
