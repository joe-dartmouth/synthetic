"""The hand-off: `python -m bko.generate --request completeness` stdout is `bko.fusion`'s input."""

from __future__ import annotations

import io
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

import bko.fusion.__main__ as fusion
import bko.generate.__main__ as generate
from bko.generate.errors import GenerationError
from tests._datasets_fixtures import pin
from tests._fusion_fixtures import UCI_X, young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import ScriptedSend, reply
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def generated(tmp_path: Path, send: ScriptedSend) -> tuple[int, Path]:
    pins, served = ladder_releases()
    uci = pin(UCI_X)
    context = generate.Context(
        open_url=RoutingOpener({**served, uci.url: UCI_X}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    out = io.StringIO()
    argv = ["per-subgroup", "--request", "completeness", "--log", str(tmp_path / "run.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    code = generate.main(argv, context=context, out=out)
    path = tmp_path / "c.out"
    path.write_text(out.getvalue(), encoding="utf-8")
    return code, path


def closed(tmp_path: Path, path: Path) -> int:
    pins, served = ladder_releases()
    uci = pin(UCI_X)
    context = fusion.Context(
        open_url=RoutingOpener({**served, uci.url: UCI_X}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
    )
    argv = [str(path), "--data-dir", str(tmp_path / "data")]
    return fusion.main(argv, context=context, out=io.StringIO())


def digest(pattern: str, caplog: pytest.LogCaptureFixture) -> str:
    [found] = [
        match.group(1)
        for record in caplog.records
        if (match := re.fullmatch(pattern, record.getMessage()))
    ]
    return found


def test_generate_stdout_is_the_closure_commands_input(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    send = ScriptedSend(reply([young(0, cp=4)] * 20), reply([young(1, cp=1)] * 20))
    code, path = generated(tmp_path, send)
    assert code == 0
    assert closed(tmp_path, path) == 0
    found = [record.getMessage() for record in caplog.records]
    assert (
        "closure definition=per-subgroup before=6 after=2 closed=4 open=2 new=0 share=0.6667"
        in found
    )
    written = digest(r"candidates rows=40 sha256=(\w+)", caplog)
    read = digest(r"candidates path=\S+ lines=40 sha256=(\w+)", caplog)
    assert read == written


def test_a_failed_generation_leaves_a_file_the_closure_command_refuses(tmp_path, caplog):
    send = ScriptedSend(
        reply([young(0, cp=4)] * 20), GenerationError("the API could not be reached")
    )
    code, path = generated(tmp_path, send)
    assert code == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 20
    assert closed(tmp_path, path) == 3
    assert any(
        record.levelno == logging.ERROR
        and "<55/female/1 has 0 lines for 20 rows" in record.getMessage()
        for record in caplog.records
    )
