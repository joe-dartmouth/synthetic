"""One real call: the API accepts the request and answers in the schema's shape.

Paid, so it runs only under `--live`. It checks structure only (keys, types and codes), never
plausibility.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from bko.datasets.cells import LABEL
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.generate.api import default_send
from bko.generate.guidance import Guidance
from bko.generate.messages import SETTINGS, Call
from bko.generate.prompt import render
from bko.ontology.frames import DIABETES_FRAME
from tests._generate_fixtures import GUIDANCE


pytestmark = pytest.mark.live

logger = logging.getLogger(__name__)

DIABETES_GUIDANCE = Guidance(
    cell=("6-7", "male", 1),
    sex=1,
    label=1,
    records=19,
    interval={
        bound.field: (float(bound.low), float(bound.high)) for bound in DIABETES_FRAME.ranges
    },
)
"""Invented guidance for E6's removed cell: the frame's own ranges, no data read."""

CASES = [
    pytest.param(UCI_FRAME, GUIDANCE, 2, 1, SETTINGS.model, id="uci-haiku"),
    pytest.param(UCI_FRAME, GUIDANCE, 2, 1, "claude-sonnet-5-5", id="uci-sonnet"),
    pytest.param(
        DIABETES_FRAME, DIABETES_GUIDANCE, SETTINGS.per_call, 10, SETTINGS.model, id="diabetes"
    ),
]
"""Each case: the frame, its guidance, the rows per call, the calls and the model."""


@pytest.mark.parametrize(("frame", "guidance", "rows", "calls", "model"), CASES)
def test_real_calls_return_schema_shaped_records(
    tmp_path, frame: FrameSpec, guidance: Guidance, rows: int, calls: int, model: str
):
    settings = dataclasses.replace(SETTINGS, model=model)
    call = Call(
        settings,
        guidance.sex,
        guidance.label,
        *render(guidance, rows, settings.guidance, frame=frame),
        frame=frame,
    )
    for index in range(calls):
        _check(tmp_path / f"call-{index}.json", call, frame, guidance, rows, model)


def _check(path, call: Call, frame: FrameSpec, guidance: Guidance, rows: int, model: str) -> None:
    reply = default_send()(call)
    path.write_text(
        json.dumps({"params": call.params(), "raw": reply.raw}, indent=2), encoding="utf-8"
    )
    logger.info(
        "request_id=%s model=%s usage=%s call=%s", reply.request_id, reply.model, reply.usage, path
    )
    assert reply.stop_reason == "end_turn"
    assert reply.text is not None
    records = json.loads(reply.text)["records"]
    assert isinstance(records, list)
    assert len(records) >= min(rows, 2)
    for record in records:
        assert set(record) == set(frame.columns)
        assert record["sex"] == guidance.sex
        assert record[LABEL] == guidance.label
        for name in frame.columns:
            value = record[name]
            assert not isinstance(value, bool), name
            assert isinstance(value, (int, float) if name in frame.fractional else int), name
        for name, codes in frame.codebook.items():
            assert record[name] in codes, name
    assert reply.attempts is not None
    assert reply.attempts >= 1
    assert reply.request_id is not None
    assert reply.request_id.startswith("req_")
    assert reply.model == model
    assert reply.usage.output_tokens <= SETTINGS.max_tokens
