"""`bko.generate.judge`: a pinned local model judges mapping candidates, tested with a fake send."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest

import bko.generate.judge as judge_module
from bko.generate.errors import GenerationError
from bko.generate.judge import (
    MODEL,
    PARTS,
    SPLIT,
    Judged,
    in_part,
    judge,
    main,
    render_verdicts,
    verdict_call,
    verdict_payload,
)
from bko.generate.local import payload
from bko.generate.messages import Call, Reply, Usage
from bko.ontology.automatic import NONE, TOP, Candidate, Option, candidates, rows_from_verdicts
from bko.ontology.terms import Ontology
from tests._automatic_fixtures import BRONCHITIS, FRAME, PNEUMONIA, automatic_ontologies


@pytest.fixture
def ontologies(tmp_path: Path) -> dict[str, Ontology]:
    return automatic_ontologies(tmp_path)


def _reply(concept: str) -> Reply:
    return Reply(
        request_id=None,
        model=MODEL,
        stop_reason="end_turn",
        text=json.dumps({"concept": concept}),
        usage=Usage(0, 0, 0, 0),
        raw={},
    )


class FakeSend:
    """Answers each call with the first option its schema offers, or `NONE` for some states."""

    def __init__(self, refuse: frozenset[str] = frozenset(), answer: str | None = None) -> None:
        self.calls: list[Call] = []
        self.refuse = refuse
        self.answer = answer

    async def __call__(self, call: Call) -> Reply:
        self.calls.append(call)
        offered = _enum(call)
        if self.answer is not None:
            return _reply(self.answer)
        return _reply(NONE if any(key in call.user for key in self.refuse) else offered[0])


def _enum(call: Call) -> list[str]:
    schema = call.reply_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    return properties["concept"]["enum"]


def _candidate(options: int) -> Candidate:
    return Candidate(
        field="glucose",
        code=1,
        meaning="high glucose",
        query="high glucose",
        options=tuple(
            Option(f"HP:{i:07d}", f"label {i}", Fraction(1, i + 1), "name", f"label {i}")
            for i in range(options)
        ),
    )


def test_the_judge_sends_temperature_0_and_a_seed():
    call = verdict_call(_candidate(3), FRAME, model=MODEL)
    body = verdict_payload(call)
    plain = payload(call)
    assert body["temperature"] == 0
    assert body["seed"] == 0
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": True, "schema": call.reply_schema()},
    }
    assert {
        k: v for k, v in body.items() if k not in {"temperature", "seed", "response_format"}
    } == {k: v for k, v in plain.items() if k not in {"temperature", "response_format"}}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_judge_offers_the_top_ten_candidates_and_none():
    call = verdict_call(_candidate(TOP + 2), FRAME, model=MODEL)
    assert TOP == 10
    assert _enum(call) == [f"HP:{i:07d}" for i in range(TOP)] + [NONE]
    schema = call.reply_schema()
    assert schema["required"] == ["concept"]
    assert schema["additionalProperties"] is False
    for i in range(TOP):
        assert f"HP:{i:07d}: label {i}" in call.user
    assert f"HP:{TOP:07d}" not in call.user
    assert "high glucose" in call.user
    assert FRAME.descriptions["glucose"] in call.user


def test_one_call_per_candidate_state(ontologies):
    send = FakeSend()
    asyncio.run(judge(FRAME, ontologies, send, model=MODEL, codes=("diag",)))
    states = candidates(FRAME, ontologies, codes=("diag",))
    assert len(send.calls) == len(states)
    assert all(call.settings.model == MODEL for call in send.calls)


def test_rows_are_a_pure_function_of_the_verdicts(ontologies):
    judged = asyncio.run(
        judge(FRAME, ontologies, FakeSend(frozenset({"chol"})), model=MODEL, codes=("diag",))
    )
    # The cross-reference state is in the table though no call asked about it.
    assert judged.verdicts[f"diag={PNEUMONIA}"] == "MONDO:0005249"
    assert judged.verdicts[f"diag={BRONCHITIS}"] == "MONDO:0005607"
    assert judged.verdicts["chol=1"] == NONE
    text = render_verdicts(judged, dataset="toy", model=MODEL)
    namespace: dict[str, object] = {}
    exec(compile(text, "<verdicts>", "exec"), namespace)  # ruff: ignore[exec-builtin]
    verdicts, labels = namespace["VERDICTS"], namespace["LABELS"]
    assert verdicts == judged.verdicts
    assert labels == judged.labels
    assert isinstance(verdicts, dict)
    assert isinstance(labels, dict)
    rows = rows_from_verdicts(FRAME, verdicts, labels)
    assert rows == rows_from_verdicts(FRAME, dict(sorted(verdicts.items())), labels)
    assert rows == rows_from_verdicts(FRAME, judged.verdicts, judged.labels)
    assert {row.concept for row in rows} == {c for c in verdicts.values() if c != NONE}
    again = asyncio.run(
        judge(FRAME, ontologies, FakeSend(frozenset({"chol"})), model=MODEL, codes=("diag",))
    )
    assert render_verdicts(again, dataset="toy", model=MODEL) == text


def test_a_split_tables_parts_partition_its_states(ontologies):
    def judged(part: str | None) -> Judged:
        send = FakeSend(frozenset({"chol"}))
        return asyncio.run(judge(FRAME, ontologies, send, model=MODEL, codes=("diag",), part=part))

    whole, low, high = judged(None), *(judged(part) for part in PARTS)
    assert PARTS == ("low", "high")
    assert low.verdicts.keys().isdisjoint(high.verdicts.keys())
    assert {**low.verdicts, **high.verdicts} == whole.verdicts
    assert {**low.labels, **high.labels} == whole.labels
    assert all(key.partition("=")[0] < SPLIT for key in low.verdicts)
    assert all(key.partition("=")[0] >= SPLIT for key in high.verdicts)
    assert low.verdicts
    assert high.verdicts
    text = render_verdicts(low, dataset="toy", model=MODEL, part="low")
    assert 'PART: Final = "low"\n' in text
    assert "PART:" not in render_verdicts(whole, dataset="toy", model=MODEL)
    with pytest.raises(ValueError, match="middle"):
        in_part("chol", "middle")


def test_a_verdict_outside_the_candidates_is_refused(ontologies):
    send = FakeSend(answer="MONDO:9999999")
    with pytest.raises(GenerationError, match="MONDO:9999999"):
        asyncio.run(judge(FRAME, ontologies, send, model=MODEL, codes=("diag",)))
    assert len(send.calls) == 1


@pytest.mark.parametrize("text", [None, "not json", "[]", '{"concept": 3}'])
def test_a_reply_that_is_not_a_verdict_is_refused(ontologies, text):
    async def send(call: Call) -> Reply:
        return dataclasses.replace(_reply(NONE), text=text)

    with pytest.raises(GenerationError, match="verdict"):
        asyncio.run(judge(FRAME, ontologies, send, model=MODEL, codes=("diag",)))


def test_the_command_writes_the_verdict_module(ontologies, monkeypatch, caplog):
    caplog.set_level("INFO")
    sent: list[object] = []

    def fake_local_send(model: str, *, url: str, build: object) -> FakeSend:
        sent.append((model, url, build))
        return FakeSend()

    monkeypatch.setattr(
        judge_module, "inputs", lambda dataset, data_dir: (FRAME, ontologies, ("diag",))
    )
    monkeypatch.setattr(judge_module, "local_send", fake_local_send)
    out = io.StringIO()
    assert main(["diabetes-codes", "--model", MODEL], out=out) == 0
    assert sent == [(MODEL, judge_module.LOCAL_URL, verdict_payload)]
    assert "VERDICTS" in out.getvalue()
    assert any(
        m.startswith(f"judge dataset=diabetes-codes model={MODEL} ") for m in caplog.messages
    )
    assert any(m.startswith("verdict field=glucose code=2 ") for m in caplog.messages)


def test_the_command_exits_two_when_the_judge_fails(ontologies, monkeypatch):
    monkeypatch.setattr(
        judge_module, "inputs", lambda dataset, data_dir: (FRAME, ontologies, ("diag",))
    )
    monkeypatch.setattr(
        judge_module, "local_send", lambda model, *, url, build: FakeSend(answer="HP:1")
    )
    assert main(["diabetes", "--model", MODEL], out=io.StringIO()) == 2
