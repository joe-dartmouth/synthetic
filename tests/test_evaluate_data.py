"""One `Data` value per named run: every stage of `score` reads its frame, records and digest."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import BinaryIO

import pytest

from bko.datasets.fetch import ensure
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.evaluate.__main__ import _parser
from bko.evaluate.context import Context
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, RUNS, NamedRun
from bko.evaluate.holdout_candidates import SCHEMA, CandidateFile, Origin, read
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.scoring import score_data
from bko.generate.errors import GenerationError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend, Call
from bko.validate.checks import validate
from tests._evaluate_bko_fixtures import NARROW, candidates_for, context
from tests._evaluate_fixtures import FAST_ARMS
from tests._frame_fixtures import TOY, toy_records
from tests._generate_fixtures import ScriptedSend
from tests._validate_fixtures import planted


def _digest(records) -> str:
    return hashlib.sha256(records.to_csv(lineterminator="\n").encode()).hexdigest()


def _toy_data() -> Data:
    records = toy_records()
    return Data(records, TOY, (), "toy", _digest(records))


def _refusing_opener(url: str, timeout: float) -> BinaryIO:
    msg = f"no download in this test: {url} ({timeout} s)"
    raise AssertionError(msg)


def _unsendable() -> AnySend:
    msg = "a rehearsal never builds the live send"
    raise AssertionError(msg)


def _toy_context(**changes: object) -> Context:
    built = Context(
        open_url=_refusing_opener,
        experiment=NARROW,
        arms={"none": FAST_ARMS["none"]},
        releases={},
        field_rows=(),
        send=_unsendable,
    )
    return dataclasses.replace(built, **changes)


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_uci_data_carries_frame_rows_name_and_digest(tmp_path):
    ctx, _ = context(ScriptedSend())
    data = load(Dataset.UCI, tmp_path, ctx)
    records = load_uci(tmp_path, source=ctx.uci)
    assert data.frame is UCI_FRAME
    assert data.field_rows == ctx.field_rows
    assert data.name == ctx.uci.name
    assert data.records.equals(records)
    assert data.digest == _digest(records)
    ensure(ctx.uci, tmp_path, open_url=_refusing_opener)


def test_score_data_rehearses_over_the_toy_frame(tmp_path):
    log = tmp_path / "calls.jsonl"
    candidates = candidates_for(log)
    argv = ["score", "toy", "--rehearse", "--log", str(log), "--candidates", str(candidates)]
    args = _parser().parse_args([*argv, "--data-dir", str(tmp_path / "data")])
    named = NamedRun(NARROW)
    text = score_data(_toy_data(), named, args, _toy_context())
    assert ",bko," in text
    kept = read(candidates.read_bytes())
    assert kept.header["frame"] == "toy"
    assert kept.header["data"] == _toy_data().digest
    assert kept.candidates
    for line in kept.candidates:
        assert isinstance(line.candidate, dict)
        assert set(line.candidate) == set(TOY.columns)
    lines = _lines(log)
    assert lines[0]["frame"] == "toy"
    assert lines[-1]["status"] == "complete"
    requests = [line for line in lines if line["kind"] == "request"]
    user = requests[0]["params"]["messages"][0]["content"]
    assert "The findings below" not in user
    assert "No record of this group" not in user
    assert not (tmp_path / "data").exists()


def test_the_direct_arm_is_refused_on_a_non_uci_frame(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    args = _parser().parse_args(
        ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    )
    named = RUNS["g2"]
    assert DIRECT_ARM in named.experiment.arms
    with pytest.raises(EvaluationError, match="direct arm"):
        score_data(_toy_data(), named, args, _toy_context(experiment=named.experiment))
    assert not candidates.exists()


def test_the_call_log_run_line_names_the_frame(tmp_path):
    with CallLog(tmp_path / "toy.jsonl", SETTINGS, frame="toy"):
        pass
    with CallLog(tmp_path / "uci.jsonl", SETTINGS):
        pass
    assert _lines(tmp_path / "toy.jsonl")[0]["frame"] == "toy"
    assert _lines(tmp_path / "uci.jsonl")[0]["frame"] == "uci"


def test_the_validated_line_names_the_frame(caplog):
    with caplog.at_level(logging.INFO):
        validate([{"age": 50, "sex": 0, "level": 1.25, "target": 1}], frame=TOY)
        validate([planted()])
    lines = [record.getMessage() for record in caplog.records if "validated" in record.message]
    assert lines[0] == "validated generated=1 passed=1 rejected=0 none frame=toy"
    assert lines[1] == "validated generated=1 passed=1 rejected=0 none frame=uci"


def test_the_candidates_header_names_frame_and_data_and_is_schema_2(tmp_path):
    data = _toy_data()
    origin = Origin("toy", data, {})
    messages = tmp_path / "messages.jsonl"
    with CandidateFile(messages, origin, SETTINGS):
        pass
    header = read(messages.read_bytes()).header
    assert SCHEMA == 2
    assert header["schema"] == 2
    assert (header["frame"], header["data"]) == ("toy", data.digest)
    assert "uci" not in header
    assert header["settings"] == json.loads(json.dumps(dataclasses.asdict(SETTINGS)))
    direct = tmp_path / "direct.jsonl"
    with CandidateFile(direct, origin, None):
        pass
    assert "settings" not in read(direct.read_bytes()).header
    old = {**header, "schema": 1}
    with pytest.raises(Exception, match="schema is 1"):
        read((json.dumps(old) + "\n").encode())
    older = {key: value for key, value in header.items() if key not in {"frame", "data"}}
    with pytest.raises(Exception, match="header"):
        read((json.dumps({**older, "uci": "ab" * 32}) + "\n").encode())


def test_a_canned_send_refusal_keeps_its_message(tmp_path):
    send = CannedSend(_toy_data(), NARROW, SETTINGS)
    call = Call(SETTINGS, 0, 1, "system", "user", frame=UCI_FRAME)
    with pytest.raises(GenerationError, match="frame uci") as caught:
        send(call)
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="toy")
    with log:
        text = log.failure(1, caught.value)
    assert text == str(caught.value)
    error = next(line for line in _lines(tmp_path / "calls.jsonl") if line["kind"] == "error")
    assert error["error"] == str(caught.value)
    assert "frame uci" in error["error"]
