"""`python -m bko.ontology check` over planted releases served by a routing opener."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import logging
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

import bko.ontology.__main__ as ontology_main
from bko.datasets.sources import Source
from bko.ontology.__main__ import Context, configure_logging, main
from bko.ontology.mapping import CODE_MAPPINGS, UNMAPPED_CODES
from bko.ontology.schema import CheckedRow, CodeMapping, Codes, FieldMapping, OutOfScope
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import pin
from tests._ontology_fixtures import (
    HPO_TEXT,
    MONDO_TEXT,
    SANTOS_DO_TEXT,
    SANTOS_MONDO_TEXT,
    SANTOS_REPORT,
    RoutingOpener,
)


FIELD_ROW = FieldMapping(
    when=(Codes("cp", frozenset({1, 2, 3})),),
    concept="HP:0100749",
    label="Chest pain",
    source="uci-heart-disease",
    limits="test",
)
SCOPE_ROW = OutOfScope(concept="HP:0000822", label="Hypertension", reason="test", source="hpo")
ROWS: tuple[CheckedRow, ...] = (*CODE_MAPPINGS, FIELD_ROW, SCOPE_ROW, *UNMAPPED_CODES)

EXPECTED = (
    "table,subject,concept,label,status\n"
    "code,SCTID:414545008,MONDO:0024644,myocardial ischemia,ok\n"
    "code,SCTID:59621000,MONDO:0001134,essential hypertension,ok\n"
    "code,SCTID:44054006,MONDO:0005148,type 2 diabetes mellitus,ok\n"
    "code,SCTID:55822004,MONDO:0021187,hyperlipidemia,ok\n"
    "code,SCTID:22298006,MONDO:0005068,myocardial infarction,ok\n"
    "code,SCTID:401303003,MONDO:0041656,ST-elevation myocardial infarction,ok\n"
    "code,SCTID:230690007,MONDO:0005098,stroke disorder,ok\n"
    "field,cp in 1|2|3,HP:0100749,Chest pain,ok\n"
    "out-of-scope,,HP:0000822,Hypertension,ok\n"
    "excluded-code,SCTID:401314000,,Acute non-ST segment elevation myocardial infarction "
    "(disorder),ok\n"
)


def _releases(mondo: bytes, hpo: bytes) -> tuple[dict[str, Source], dict[str, bytes]]:
    mondo_pin, hpo_pin = pin(mondo, name="mondo"), pin(hpo, name="hpo")
    return {"HP": hpo_pin, "MONDO": mondo_pin}, {mondo_pin.url: mondo, hpo_pin.url: hpo}


def _run(
    data_dir: Path,
    *,
    rows: tuple[CheckedRow, ...] = ROWS,
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
    assert "terms=29 obsolete=2" in caplog.text
    assert "checked 10 rows: 0 disagree" in caplog.text
    assert f"parser pronto {importlib.metadata.version('pronto')}" in caplog.text


def _excluded_rows(out: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(out)) if row[0] == "excluded-code"]


def test_the_default_rows_include_the_excluded_codes(tmp_path: Path) -> None:
    releases, bodies = _releases(MONDO_TEXT.encode(), HPO_TEXT.encode())
    out = io.StringIO()
    context = Context(open_url=RoutingOpener(bodies), releases=releases)
    code = main(["check", "--data-dir", str(tmp_path / "data")], context=context, out=out)
    assert code == 0
    assert _excluded_rows(out.getvalue()) == [
        [
            "excluded-code",
            "SCTID:401314000",
            "",
            "Acute non-ST segment elevation myocardial infarction (disorder)",
            "ok",
        ]
    ]


def test_a_second_run_is_identical_and_offline(tmp_path):
    _, first, opener = _run(tmp_path / "data")
    calls = len(opener.calls)
    _, second, _ = _run(tmp_path / "data", opener=opener)
    assert second == first
    assert len(opener.calls) == calls


def test_a_disagreeing_row_exits_four_after_the_report(tmp_path, caplog):
    wrong = CodeMapping(
        snomed="414545008", concept="MONDO:0024644", label="Myocardial Ischemia", source="mondo"
    )
    with caplog.at_level(logging.WARNING, logger="bko.ontology"):
        code, out, _ = _run(tmp_path / "data", rows=(wrong, FIELD_ROW, SCOPE_ROW))
    assert code == 4
    lines = out.splitlines()
    assert len(lines) == 4
    assert lines[1].split(",")[-1].startswith("named 'myocardial ischemia'")
    assert lines[2:] == [
        line for line in EXPECTED.splitlines() if line.startswith(("field,", "out-of-scope,"))
    ]
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.startswith("row code MONDO:0024644 disagrees: named 'myocardial ischemia'")


def test_an_excluded_code_that_gains_a_match_warns_with_its_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stanza = "id: MONDO:0005068\nname: myocardial infarction\n"
    assert stanza in MONDO_TEXT
    gained = MONDO_TEXT.replace(
        stanza, f"{stanza}property_value: skos:exactMatch SCTID:401314000\n"
    )
    with caplog.at_level(logging.WARNING, logger="bko.ontology"):
        code, out, _ = _run(tmp_path / "data", rows=UNMAPPED_CODES, mondo=gained.encode())
    assert code == 4
    [row] = _excluded_rows(out)
    assert row[-1] == "skos:exactMatch SCTID:401314000 is on MONDO:0005068"
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warning == (
        "row excluded-code SCTID:401314000 disagrees: "
        "skos:exactMatch SCTID:401314000 is on MONDO:0005068"
    )


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


def test_configure_logging_calls_the_shared_setup(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(ontology_main, "configure", lambda **kwargs: calls.append(kwargs))
    configure_logging()
    assert calls == [{}]


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
    assert "{check,reproduce,elk}" in result.stdout


def _reproduce(
    data_dir: Path, *, do: bytes = SANTOS_DO_TEXT.encode(), opener: RoutingOpener | None = None
) -> tuple[int, str, RoutingOpener]:
    mondo = SANTOS_MONDO_TEXT.encode()
    mondo_pin, do_pin = pin(mondo, name="mondo"), pin(do, name="do")
    opener = opener or RoutingOpener({mondo_pin.url: mondo, do_pin.url: do})
    out = io.StringIO()
    context = Context(open_url=opener, paper_releases={"DOID": do_pin, "MONDO": mondo_pin})
    code = main(["reproduce", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue(), opener


def test_reproduce_writes_the_report_and_exits_zero(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.ontology"):
        code, out, opener = _reproduce(tmp_path / "data")
    assert (code, out) == (0, SANTOS_REPORT)
    assert len(opener.calls) == 2
    assert "release do data_version=releases/2026-09-30/doid.obo" in caplog.text
    assert f"parser pronto {importlib.metadata.version('pronto')}" in caplog.text
    assert "reproduced 40 findings: 0 differ" in caplog.text


def test_reproduce_warns_per_differing_finding_and_still_exits_zero(tmp_path, caplog):
    do = (SANTOS_DO_TEXT + "\n[Term]\nid: DOID:114466\nname: sciatic neuropathy\n").encode()
    with caplog.at_level(logging.WARNING, logger="bko.ontology"):
        code, out, _ = _reproduce(tmp_path / "data", do=do)
    assert code == 0
    row = "term,DOID:114466,Sciatic Neuropathy,DOID:114466 present; DOID:11446 sciatic neuropathy"
    assert f"{row},differs" in out.splitlines()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        (
            "finding term DOID:114466 differs: paper 'Sciatic Neuropathy', "
            "measured 'DOID:114466 present; DOID:11446 sciatic neuropathy'"
        )
    ]


def test_reproduce_with_an_unreachable_release_exits_two(tmp_path):
    opener = RoutingOpener({}, error=urllib.error.URLError("down"))
    code, out, _ = _reproduce(tmp_path / "data", opener=opener)
    assert (code, out) == (2, "")


def test_reproduce_with_a_changed_release_exits_three(tmp_path):
    mondo_pin = pin(SANTOS_MONDO_TEXT.encode(), name="mondo")
    do = SANTOS_DO_TEXT.encode()
    served = {mondo_pin.url: b"not the pinned bytes", pin(do, name="do").url: do}
    code, out, _ = _reproduce(tmp_path / "data", opener=RoutingOpener(served))
    assert (code, out) == (3, "")
