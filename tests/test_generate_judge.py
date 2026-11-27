"""`bko.generate.judge`: a pinned local model judges mapping candidates, tested with a fake send."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest

import bko.generate.judge as judge_module
from bko.datasets.errors import ContractError
from bko.generate.errors import GenerationError
from bko.generate.judge import (
    MODEL,
    PARTS,
    SPLIT,
    SYSTEM,
    Judged,
    in_part,
    judge,
    main,
    read_verdict,
    render_verdicts,
    verdict_call,
    verdict_payload,
)
from bko.generate.local import payload
from bko.generate.messages import Call, Reply, Usage
from bko.ontology.automatic import NONE, TOP, Candidate, candidates, rows_from_verdicts
from bko.ontology.lexical import Option
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


def _command(monkeypatch, ontologies, send: FakeSend, checked: list[str] | None = None) -> None:
    """Point `main` at the fixture frame, a fake server check and `send`."""
    monkeypatch.setattr(
        judge_module, "inputs", lambda dataset, data_dir: (FRAME, ontologies, ("diag",))
    )
    monkeypatch.setattr(
        judge_module,
        "check_server",
        lambda model, port: (checked if checked is not None else []).append(model),
    )
    monkeypatch.setattr(judge_module, "local_send", lambda model, *, url, build: send)


def test_the_command_writes_the_verdict_module(ontologies, monkeypatch, caplog):
    caplog.set_level("INFO")
    sent: list[object] = []

    def fake_local_send(model: str, *, url: str, build: object) -> FakeSend:
        sent.append((model, url, build))
        return FakeSend()

    _command(monkeypatch, ontologies, FakeSend())
    monkeypatch.setattr(judge_module, "local_send", fake_local_send)
    out = io.BytesIO()
    assert main(["diabetes-codes"], out=out) == 0
    assert sent == [(MODEL, judge_module.LOCAL_URL, verdict_payload)]
    data = out.getvalue()
    assert b"VERDICTS" in data
    calls = len(candidates(FRAME, ontologies, codes=("diag",)))
    assert f"judge start dataset=diabetes-codes part=all calls={calls}" in caplog.messages
    assert any(m.startswith(f"verdict i=1/{calls} field=") for m in caplog.messages)
    assert any(" seconds=" in m for m in caplog.messages if m.startswith("verdict "))
    (closing,) = [m for m in caplog.messages if m.startswith("judge dataset=")]
    assert f" calls={calls} elapsed=" in closing
    assert closing.endswith(f"sha256={hashlib.sha256(data).hexdigest()}")


def test_the_command_checks_the_server_before_reading_the_releases(ontologies, monkeypatch):
    order: list[str] = []

    def refuse(model: str, port: int) -> None:
        order.append("check")
        raise GenerationError("the local server serves another alias")

    monkeypatch.setattr(judge_module, "check_server", refuse)
    monkeypatch.setattr(judge_module, "inputs", lambda dataset, data_dir: order.append("inputs"))
    assert main(["diabetes"], out=io.BytesIO()) == 2
    assert order == ["check"]


def test_the_command_exits_two_when_the_judge_fails(ontologies, monkeypatch):
    _command(monkeypatch, ontologies, FakeSend(answer="HP:1"))
    out = io.BytesIO()
    assert main(["diabetes", "--model", MODEL], out=out) == 2
    assert out.getvalue() == b""


def test_the_command_returns_a_dataset_errors_code(monkeypatch):
    def refuse(dataset: str, data_dir: Path | None) -> None:
        raise ContractError("the archive contradicts its pin")

    monkeypatch.setattr(judge_module, "check_server", lambda model, port: None)
    monkeypatch.setattr(judge_module, "inputs", refuse)
    assert main(["diabetes"], out=io.BytesIO()) == ContractError.exit_code == 3


def test_a_run_with_no_candidate_is_refused(ontologies, monkeypatch, tmp_path):
    frame = dataclasses.replace(
        FRAME,
        features=("age", "sex"),
        categorical=("sex",),
        codebook={"sex": FRAME.codebook["sex"]},
        descriptions={"age": "a", "sex": "sex", "target": "t"},
        label_meanings={0: "no", 1: "zzz"},
    )
    with pytest.raises(ValueError, match="no candidate state"):
        asyncio.run(judge(frame, ontologies, FakeSend(), model=MODEL))
    monkeypatch.setattr(judge_module, "inputs", lambda dataset, data_dir: (frame, ontologies, ()))
    monkeypatch.setattr(judge_module, "check_server", lambda model, port: None)
    assert main(["diabetes"], out=io.BytesIO()) == 2


def test_a_part_is_only_for_the_split_dataset_and_a_thinking_model_is_refused():
    for argv in (["diabetes", "--part", "low"], ["diabetes", "--model", f"{MODEL}-thinking"]):
        with pytest.raises(SystemExit) as exited:
            main(argv, out=io.BytesIO())
        assert exited.value.code == 2


def test_a_failed_call_names_its_candidate_and_place(ontologies):
    async def send(call: Call) -> Reply:
        raise GenerationError("the server closed the connection", retryable=True)

    states = candidates(FRAME, ontologies, codes=("diag",))
    with pytest.raises(GenerationError, match=rf"{states[0].key} \(1/{len(states)}\)") as raised:
        asyncio.run(judge(FRAME, ontologies, send, model=MODEL, codes=("diag",)))
    assert raised.value.retryable


def test_the_split_field_opens_the_high_part():
    assert in_part(SPLIT, "high")
    assert not in_part(SPLIT, "low")
    assert in_part("diag_2", "low")


def test_only_the_verdict_concepts_label_is_recorded(ontologies):
    judged = asyncio.run(judge(FRAME, ontologies, FakeSend(), model=MODEL, codes=("diag",)))
    offered = {o.curie for c in candidates(FRAME, ontologies, codes=("diag",)) for o in c.options}
    chosen = {c for c in judged.verdicts.values() if c != NONE}
    assert set(judged.labels) == chosen
    assert offered - chosen


PAYLOAD_SHA256 = "943d2236b0497b24470d6a6d0db9a6b48ef0de8120d70b8522c999a7f7c97eb3"
"""The sha256 of a three-option verdict call's request body, as `local_send` posts it."""

SYSTEM_SHA256 = "74d84b79b97e5ca61278d8d7bf1257b7959471e9c5f2b2cb0e6202cf4d7bc6e5"
"""The sha256 of the verdict calls' system turn."""


def test_the_verdict_request_bytes_and_system_turn_are_pinned():
    body = verdict_payload(verdict_call(_candidate(3), FRAME, model=MODEL))
    assert hashlib.sha256(json.dumps(body).encode()).hexdigest() == PAYLOAD_SHA256
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA256


def test_the_eleventh_option_is_refused():
    candidate = _candidate(TOP + 2)
    beyond = candidate.options[TOP].curie
    with pytest.raises(GenerationError, match=beyond):
        read_verdict(_reply(beyond), candidate)
    assert read_verdict(_reply(candidate.options[TOP - 1].curie), candidate)
