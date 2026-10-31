"""One `Data` value per named run: every stage of `score` reads its frame, records and digest."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import BinaryIO, cast

import pytest

from bko.datasets.fetch import ensure
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.evaluate.__main__ import _parser
from bko.evaluate.context import Context
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import CandidatesFormatError, EvaluationError
from bko.evaluate.experiment import DIRECT_ARM
from bko.evaluate.holdout_candidates import SCHEMA, CandidateFile, Origin, read
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.scoring import _prepare, _score, score
from bko.generate.errors import GenerationError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend, Call
from bko.validate.checks import failures, validate
from tests._evaluate_bko_fixtures import NARROW, candidates_for, context
from tests._evaluate_fixtures import FAST_ARMS
from tests._frame_fixtures import SHORT, TOY, toy_records
from tests._generate_fixtures import ScriptedSend
from tests._validate_fixtures import planted


def _digest(records) -> str:
    return hashlib.sha256(records.to_csv(lineterminator="\n").encode()).hexdigest()


def _toy_data() -> Data:
    records = toy_records()
    return Data(records, TOY, (), "toy", None)


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


TOY_BANDS = frozenset({"<45", "45-69", ">=70"})
"""The toy frame's age-band labels under its edges (45, 70)."""


def _scored(data: Data, named: NamedRun, args, ctx: Context) -> str:
    """Score `data` as `score` does once it has loaded it: `_prepare`, then `_score`."""
    return _score(data, named, args, ctx, _prepare(args, named, ctx))


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_uci_data_carries_frame_rows_name_and_digest(tmp_path, caplog):
    ctx, _ = context(ScriptedSend())
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.UCI, tmp_path, ctx)
    records = load_uci(tmp_path, source=ctx.uci)
    assert data.frame is UCI_FRAME
    assert data.field_rows == ctx.field_rows
    assert data.name == ctx.uci.name
    assert data.records.equals(records)
    assert data.digest == _digest(records)
    line = f"data dataset=uci frame=uci records={len(records)} digest={_digest(records)}"
    assert line in [record.getMessage() for record in caplog.records]
    ensure(ctx.uci, tmp_path, open_url=_refusing_opener)


def test_a_datas_digest_is_computed_from_its_records():
    records = toy_records()
    assert Data(records, TOY, (), "toy", None).digest == _digest(records)
    with pytest.raises(TypeError):
        Data(records, TOY, (), "toy", None, "ab" * 32)  # ty: ignore[too-many-positional-arguments]


def test_load_refuses_a_dataset_it_does_not_know(tmp_path):
    ctx, _ = context(ScriptedSend())
    with pytest.raises(AssertionError):
        load(cast("Dataset", "toy"), tmp_path, ctx)


def test_a_toy_rehearsal_reads_the_toy_frame_at_every_stage(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    log = tmp_path / "calls.jsonl"
    candidates = candidates_for(log)
    argv = ["score", "toy", "--rehearse", "--log", str(log), "--candidates", str(candidates)]
    args = _parser().parse_args([*argv, "--data-dir", str(tmp_path / "data")])
    named = NamedRun(NARROW)
    text = _scored(_toy_data(), named, args, _toy_context())
    assert ",bko," in text
    assert any(
        record.getMessage().startswith("experiment run=toy frame=toy ") for record in caplog.records
    )
    kept = read(candidates.read_bytes())
    assert kept.header["frame"] == "toy"
    assert kept.header["data"] == _toy_data().digest
    assert kept.candidates
    for line in kept.candidates:
        assert isinstance(line.candidate, dict)
        assert set(line.candidate) == set(TOY.columns)
        assert line.cell.split("/")[0] in TOY_BANDS
    lines = _lines(log)
    assert lines[0]["frame"] == "toy"
    assert lines[-1]["status"] == "complete"
    requests = [line for line in lines if line["kind"] == "request"]
    assert {line["cell"].split("/")[0] for line in requests} <= TOY_BANDS
    assert {"/".join(map(str, cell)) for cell in SHORT} <= {line["cell"] for line in requests}
    system = requests[0]["params"]["system"]
    assert f"a study of {TOY.study}. Each record is one {TOY.cohort}." in system
    user = requests[0]["params"]["messages"][0]["content"]
    assert "The findings below" not in user
    assert "No record of this group" not in user
    assert not (tmp_path / "data").exists()


def test_a_method_only_toy_run_asks_for_the_toy_cells(tmp_path):
    asked = []
    smote = FAST_ARMS["smote"]

    def spying(train, request):
        asked.append(request)
        return smote(train, request)

    experiment = dataclasses.replace(
        NARROW, arms=("none", "smote"), candidate="smote", comparators=("none",)
    )
    ctx = _toy_context(experiment=experiment, arms={"none": FAST_ARMS["none"], "smote": spying})
    args = _parser().parse_args(["score", "toy", "--data-dir", str(tmp_path / "data")])
    text = _scored(_toy_data(), NamedRun(experiment), args, ctx)
    assert ",smote," in text
    assert asked
    assert {request.frame.name for request in asked} == {"toy"}
    cells = {cell for request in asked for cell in request.quota}
    assert set(SHORT) <= cells
    assert {cell[0] for cell in cells} <= TOY_BANDS


def test_the_direct_arm_is_refused_on_a_non_uci_dataset_before_any_load(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    args = _parser().parse_args(
        ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    )
    named = RUNS["g2"]
    assert DIRECT_ARM in named.experiment.arms
    other = cast("Dataset", "toy")
    with pytest.raises(EvaluationError, match="direct arm"):
        score(args, _toy_context(experiment=named.experiment, dataset=other))
    assert not candidates.exists()
    assert not (tmp_path / "data").exists()


def test_the_call_log_run_line_names_the_frame(tmp_path):
    with CallLog(tmp_path / "toy.jsonl", SETTINGS, frame="toy"):
        pass
    assert _lines(tmp_path / "toy.jsonl")[0]["frame"] == "toy"
    with pytest.raises(TypeError, match="frame"):
        CallLog(tmp_path / "unnamed.jsonl", SETTINGS)  # ty: ignore[missing-argument]
    assert not (tmp_path / "unnamed.jsonl").exists()


def test_the_validated_line_names_the_frame(caplog):
    with caplog.at_level(logging.INFO):
        validate([{"age": 50, "sex": 0, "level": 1.25, "target": 1}], frame=TOY)
        validate([planted()], frame=UCI_FRAME)
        validate([], frame=TOY)
    lines = [record.getMessage() for record in caplog.records if "validated" in record.message]
    assert lines == [
        "validated generated=1 passed=1 rejected=0 none frame=toy",
        "validated generated=1 passed=1 rejected=0 none frame=uci",
        "validated no candidates frame=toy, so validity is undefined",
    ]


@pytest.mark.parametrize("check", [validate, failures])
def test_the_validate_entry_points_require_the_frame(check):
    with pytest.raises(TypeError, match="frame"):
        check([planted()])


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
    older = {key: value for key, value in header.items() if key not in {"frame", "data"}}
    schema_1 = {**older, "schema": 1, "uci": "ab" * 32}
    with pytest.raises(CandidatesFormatError, match="uci digest"):
        read((json.dumps(schema_1) + "\n").encode())
    with pytest.raises(CandidatesFormatError, match=r"schema is 3, not one of \[1, 2\]"):
        read((json.dumps({**header, "schema": 3}) + "\n").encode())
    with pytest.raises(CandidatesFormatError, match="header"):
        read((json.dumps({**older, "uci": "ab" * 32}) + "\n").encode())
    for missing in ("frame", "data"):
        partial = {key: value for key, value in header.items() if key != missing}
        with pytest.raises(CandidatesFormatError, match="header"):
            read((json.dumps(partial) + "\n").encode())


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


def test_a_canned_send_refuses_a_user_turn_it_cannot_read():
    send = CannedSend(_toy_data(), NARROW, SETTINGS)
    call = Call(SETTINGS, 0, 1, "system", "Write some records.", frame=TOY)
    with pytest.raises(GenerationError, match="does not read as the prompt renders it"):
        send(call)
    assert send.calls == 0
