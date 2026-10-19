"""`python -m bko.generate` over an archive and releases built in tmp_path, through a fake send."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bko.datasets.split import split
from bko.datasets.uci import load_uci
from bko.gaps.definitions import Definition
from bko.gaps.observe import cell_key
from bko.generate.__main__ import Context, configure_logging, main
from bko.generate.errors import GenerationError
from bko.generate.guidance import Basis
from bko.generate.messages import Call, Level, Reply, Send
from bko.generate.prompt import BASES
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import archive, cleveland_line, pin
from tests._gaps_fixtures import ANGINA, LADDER_HPO, LADDER_MONDO, STENOSIS
from tests._generate_fixtures import ScriptedSend, reply
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SIZES = {
    (40, 0, 0): 10,
    (40, 0, 1): 0,
    (40, 1, 0): 25,
    (40, 1, 1): 25,
    (63, 0, 0): 25,
    (63, 0, 1): 25,
    (63, 1, 0): 25,
    (63, 1, 1): 25,
}
"""Records per (age, sex, num). One in five of each cell is held out, so the training split holds
twenty in every cell but two: eight young women without the condition, and no young woman with
it."""

UCI = archive(
    "\n".join(
        cleveland_line(age=age, sex=sex, num=num, cp="4.0")
        for (age, sex, num), size in SIZES.items()
        for _ in range(size)
    )
    + "\n"
)

REQUEST = {("<55", "female", 0): 12, ("<55", "female", 1): 20}
"""The count-based request: each flagged cell asked for its shortfall below twenty."""

RECORD = {"age": 41, "sex": 0}
"""A decoded item; the command passes items through unchecked."""

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def releases() -> tuple[dict, dict[str, bytes]]:
    bodies = {"HP": LADDER_HPO.encode(), "MONDO": LADDER_MONDO.encode()}
    pins = {"HP": pin(bodies["HP"], name="hpo"), "MONDO": pin(bodies["MONDO"], name="mondo")}
    return pins, {pins[prefix].url: body for prefix, body in bodies.items()}


def make_context(send: Send, **change) -> Context:
    pins, served = releases()
    uci = pin(UCI)
    context = Context(
        open_url=RoutingOpener({**served, uci.url: UCI}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
        now=lambda: NOW,
    )
    return dataclasses.replace(context, **change)


def two_replies() -> ScriptedSend:
    return ScriptedSend(reply([RECORD] * 12), reply([RECORD] * 19))


def run(tmp_path: Path, context: Context, *extra: str) -> tuple[int, str, Path]:
    log = tmp_path / "run.jsonl"
    out = io.StringIO()
    argv = ["per-subgroup", "--log", str(log), "--data-dir", str(tmp_path / "data"), *extra]
    code = main(argv, context=context, out=out)
    return code, out.getvalue(), log


def log_lines(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def asked_per_cell(log: Path) -> dict[str, int]:
    asked: dict[str, int] = {}
    for line in log_lines(log):
        if line["kind"] == "request":
            cell, rows = line["cell"], line["asked"]
            assert isinstance(cell, str)
            assert isinstance(rows, int)
            asked[cell] = asked.get(cell, 0) + rows
    return asked


def test_the_command_asks_each_flagged_cell_for_its_shortfall_and_prints_one_line_per_candidate(
    tmp_path,
):
    send = two_replies()
    code, out, log = run(tmp_path, make_context(send))
    assert code == 0
    assert asked_per_cell(log) == {"<55/female/0": 12, "<55/female/1": 20}
    assert [(call.sex, call.label) for call in send.calls] == [(0, 0), (0, 1)]
    assert len(out.splitlines()) == sum(REQUEST.values())
    send.assert_consumed()


def test_the_request_equals_e1s_gap_request(tmp_path):
    from bko.evaluate.experiment import E1, gap_request

    code, _, log = run(tmp_path, make_context(two_replies()))
    assert code == 0
    train = split(load_uci(tmp_path / "data", source=pin(UCI))).train
    expected = gap_request(train, E1, seed=0).quota
    assert asked_per_cell(log) == {cell_key(cell): rows for cell, rows in expected.items()}
    assert dict(expected) == REQUEST


@pytest.mark.parametrize(
    ("extra", "level"), [((), Level.ONTOLOGY), (("--guidance", "cell"), Level.CELL)]
)
def test_the_guidance_flag_sets_the_level_and_defaults_to_ontology(tmp_path, extra, level):
    send = two_replies()
    code, _, log = run(tmp_path, make_context(send), *extra)
    assert code == 0
    assert {call.settings.guidance for call in send.calls} == {level}
    assert log_lines(log)[0]["settings"]["guidance"] == level.value


OLDER_MEN_ANGINA = archive(
    "\n".join(
        cleveland_line(age=age, sex=sex, num=num, cp="1.0" if (age, sex) == (63, 1) else "4.0")
        for (age, sex, num), size in SIZES.items()
        for _ in range(size)
    )
    + "\n"
)
"""`UCI` with typical angina in every older man, so the training split shows it but no young
woman does: a per-subgroup gap of each young-women cell, no population-level gap, and a
frequency-aware gap of the young women without the condition."""


@pytest.mark.parametrize(
    ("definition", "basis"),
    [
        (Definition.SUBGROUP, [Basis.ABSENT]),
        (Definition.POPULATION, []),
        (Definition.FREQUENCY, [Basis.RARE]),
    ],
)
def test_the_definition_chooses_the_gaps_the_prompt_states(tmp_path, definition, basis):
    _, served = releases()
    uci = pin(OLDER_MEN_ANGINA)
    send = two_replies()
    context = make_context(
        send, open_url=RoutingOpener({**served, uci.url: OLDER_MEN_ANGINA}), uci=uci
    )
    argv = [definition.value, "--log", str(tmp_path / "run.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=context, out=io.StringIO()) == 0
    first = send.calls[0]
    assert (first.sex, first.label) == (0, 0)
    assert [key for key, sentence in BASES.items() if sentence in first.user] == basis
    send.assert_consumed()


def test_an_existing_log_path_exits_2_before_any_download_or_call(tmp_path, caplog):
    log = tmp_path / "run.jsonl"
    log.write_text("paid\n", encoding="utf-8")
    send = ScriptedSend()
    code, out, _ = run(tmp_path, make_context(send))
    assert (code, out) == (2, "")
    assert send.calls == []
    assert log.read_text(encoding="utf-8") == "paid\n"
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert not (tmp_path / "data").exists(), "the log was checked only after the downloads"


def test_an_api_failure_exits_2_keeping_earlier_batches_on_stdout_and_the_spend_on_stderr(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO)
    send = ScriptedSend(reply([RECORD] * 12), GenerationError("the API could not be reached"))
    code, out, log = run(tmp_path, make_context(send))
    assert code == 2
    assert [json.loads(line)["call"] for line in out.splitlines()] == [1] * 12
    send.assert_consumed()
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert f"candidates rows=12 sha256={digest}" in [r.getMessage() for r in caplog.records]
    [summary] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("run ")]
    assert "status=failed calls=2 failed=1" in summary
    assert "usd=0.0140" in summary
    assert log_lines(log)[-1]["status"] == "failed"
    assert any(
        r.levelno == logging.ERROR and r.getMessage() == "the API could not be reached"
        for r in caplog.records
    )


def test_a_dataset_contract_error_exits_3(tmp_path):
    tampered = bytes([UCI[0] ^ 1]) + UCI[1:]
    _, served = releases()
    uci = pin(UCI)
    send = ScriptedSend()
    context = make_context(send, open_url=RoutingOpener({**served, uci.url: tampered}), uci=uci)
    code, out, log = run(tmp_path, context)
    assert (code, out) == (3, "")
    assert send.calls == []
    assert not log.exists()


class FlushCounter(io.StringIO):
    """A stdout that counts its flushes."""

    flushes = 0

    def flush(self) -> None:
        """Count the flush."""
        self.flushes += 1
        super().flush()


def test_each_batch_is_written_and_flushed_as_its_call_returns(tmp_path):
    out = FlushCounter()
    seen: list[tuple[int, int]] = []
    scripted = two_replies()

    def send(call: Call) -> Reply:
        seen.append((len(out.getvalue().splitlines()), out.flushes))
        return scripted(call)

    argv = ["per-subgroup", "--log", str(tmp_path / "run.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=make_context(send), out=out) == 0
    assert seen[0] == (0, 0)
    assert seen[1][0] == 12
    assert seen[1][1] >= 1


def test_stdout_is_json_lines_of_call_cell_position_and_candidate_or_undecoded_reason(tmp_path):
    _, out, _ = run(tmp_path, make_context(two_replies()))
    lines = [json.loads(line) for line in out.splitlines()]
    assert lines[0] == {"call": 1, "cell": "<55/female/0", "position": 0, "candidate": RECORD}
    assert lines[12] == {"call": 2, "cell": "<55/female/1", "position": 0, "candidate": RECORD}
    assert lines[-1] == {"call": 2, "cell": "<55/female/1", "position": 19, "undecoded": "short"}
    assert all(line == json.dumps(json.loads(line), sort_keys=True) for line in out.splitlines())


def test_the_summary_line_names_calls_tokens_usd_log_and_stdout_sha256(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _, out, log = run(tmp_path, make_context(two_replies()))
    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"run status=complete calls=2 failed=0 asked=32 decoded=31 undecoded=1 "
        f"input_tokens=2000 output_tokens=1000 usd=0.0280 log={log}"
    ) in messages
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert messages[-1] == f"candidates rows=32 sha256={digest}"


def test_configure_logging_sends_info_to_stderr_and_quiets_the_http_client(monkeypatch):
    """`captureWarnings` is stubbed: the real call is process-wide and outlives the test."""
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(logging, "captureWarnings", lambda _: None)
    quiet = logging.getLogger("httpx2")
    monkeypatch.setattr(quiet, "level", quiet.level)
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO
    assert quiet.level == logging.WARNING


def test_the_module_runs_as_a_program():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
    result = subprocess.run(
        [sys.executable, "-m", "bko.generate", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--guidance" in result.stdout
    assert "--log" in result.stdout
