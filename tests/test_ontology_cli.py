"""`python -m bko.ontology check` over planted releases served by a routing opener."""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from bko.datasets.sources import Source
from bko.ontology.__main__ import Context, configure_logging, main
from bko.ontology.mapping import CODE_MAPPINGS
from bko.ontology.schema import CodeMapping, Codes, ConceptRow, FieldMapping, OutOfScope
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import pin
from tests._ontology_fixtures import HPO_TEXT, MONDO_TEXT, RoutingOpener


FIELD_ROW = FieldMapping(
    when=(Codes("cp", frozenset({1, 2, 3})),),
    concept="HP:0100749",
    label="Chest pain",
    source="uci-heart-disease",
    limits="test",
)
SCOPE_ROW = OutOfScope(concept="HP:0000822", label="Hypertension", reason="test", source="hpo")
ROWS: tuple[ConceptRow, ...] = (*CODE_MAPPINGS, FIELD_ROW, SCOPE_ROW)

EXPECTED = (
    "table,subject,concept,label,status\n"
    "code,SCTID:414545008,MONDO:0024644,myocardial ischemia,ok\n"
    "field,cp in 1|2|3,HP:0100749,Chest pain,ok\n"
    "out-of-scope,,HP:0000822,Hypertension,ok\n"
)


def _releases(mondo: bytes, hpo: bytes) -> tuple[dict[str, Source], dict[str, bytes]]:
    mondo_pin, hpo_pin = pin(mondo, name="mondo"), pin(hpo, name="hpo")
    return {"HP": hpo_pin, "MONDO": mondo_pin}, {mondo_pin.url: mondo, hpo_pin.url: hpo}


def _run(
    data_dir: Path,
    *,
    rows: tuple[ConceptRow, ...] = ROWS,
    mondo: bytes = MONDO_TEXT.encode(),
    hpo: bytes = HPO_TEXT.encode(),
    served: dict[str, bytes] | None = None,
    opener: RoutingOpener | None = None,
) -> tuple[int, str, RoutingOpener]:
    releases, bodies = _releases(mondo, hpo)
    if served is not None:
        bodies = {url: served.get(url, body) for url, body in bodies.items()}
    opener = opener or RoutingOpener(bodies)
    out = io.StringIO()
    context = Context(open_url=opener, releases=releases, rows=rows)
    code = main(["check", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue(), opener


def test_agreeing_rows_exit_zero(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code, out, _ = _run(tmp_path / "data")
    assert code == 0
    assert out == EXPECTED
    assert "data_version=releases/2026-10-06" in caplog.text
    assert "data_version=hp/releases/2026-09-01" in caplog.text
    assert "release mondo data_version=releases/2026-10-06" in caplog.text
    assert "terms=9 obsolete=2" in caplog.text
    assert "checked 3 rows: 0 disagree" in caplog.text


def test_a_second_run_is_identical_and_offline(tmp_path):
    _, first, opener = _run(tmp_path / "data")
    calls = len(opener.calls)
    _, second, _ = _run(tmp_path / "data", opener=opener)
    assert second == first
    assert len(opener.calls) == calls


def test_a_disagreeing_row_exits_three_after_the_report(tmp_path):
    wrong = CodeMapping(
        snomed="414545008", concept="MONDO:0024644", label="Myocardial Ischemia", source="mondo"
    )
    code, out, _ = _run(tmp_path / "data", rows=(wrong, FIELD_ROW, SCOPE_ROW))
    assert code == 3
    lines = out.splitlines()
    assert len(lines) == 4
    assert lines[1].split(",")[-1].startswith("named 'myocardial ischemia'")
    assert lines[2:] == EXPECTED.splitlines()[2:]


def test_an_unreachable_release_exits_two(tmp_path):
    opener = RoutingOpener({}, error=urllib.error.URLError("down"))
    code, out, _ = _run(tmp_path / "data", opener=opener)
    assert (code, out) == (2, "")


def test_a_changed_body_exits_three(tmp_path):
    releases, _ = _releases(MONDO_TEXT.encode(), HPO_TEXT.encode())
    changed = bytearray(MONDO_TEXT.encode())
    changed[0] ^= 1
    served = {releases["MONDO"].url: bytes(changed)}
    code, out, _ = _run(tmp_path / "data", served=served)
    assert (code, out) == (3, "")


def test_a_release_without_data_version_exits_three(tmp_path):
    mondo = MONDO_TEXT.replace("data-version: releases/2026-10-06\n", "").encode()
    code, out, _ = _run(tmp_path / "data", mondo=mondo)
    assert (code, out) == (3, "")


def test_an_unwritable_data_root_exits_two(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    if os.access(locked, os.W_OK):
        locked.chmod(0o700)
        pytest.skip("this user can write to a read-only directory, as root can")
    try:
        code, out, _ = _run(locked / "data")
    finally:
        locked.chmod(0o700)
    assert (code, out) == (2, "")


def test_configure_logging_sends_info_to_stderr(monkeypatch):
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO


def test_the_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "bko.ontology", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "check" in result.stdout
