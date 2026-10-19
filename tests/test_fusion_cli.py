"""`python -m bko.fusion` over an archive, releases and a candidates file built in tmp_path."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import bko.gaps.__main__ as gaps_command
from bko.datasets.sources import Source
from bko.datasets.split import split
from bko.datasets.uci import load_uci
from bko.fusion.__main__ import Context, configure_logging, main
from bko.gaps.definitions import Definition
from bko.generate.decode import Reason, Undecoded
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import archive, cleveland_line, pin
from tests._fusion_fixtures import UCI_X, candidate_lines, scenario_a, training_split, young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import planted


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

HEADER = (
    "definition,source,antecedent,antecedent_label,consequent,consequent_label,"
    "age_band,sex,target,before,after"
)


def make_context(data: bytes = UCI_X, **change) -> Context:
    pins, served = ladder_releases()
    uci: Source = pin(data)
    context = Context(
        open_url=RoutingOpener({**served, uci.url: data}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
    )
    return dataclasses.replace(context, **change)


def run(tmp_path: Path, candidates: bytes, context: Context | None = None) -> tuple[int, str, Path]:
    path = tmp_path / "c.out"
    path.write_bytes(candidates)
    out = io.StringIO()
    argv = [str(path), "--data-dir", str(tmp_path / "data")]
    code = main(argv, context=context or make_context(), out=out)
    return code, out.getvalue(), path


def rows_of(out: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(out)))[1:]


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_the_report_lists_each_definitions_pairs_before_and_after_the_join(tmp_path):
    code, out, _ = run(tmp_path, scenario_a())
    assert code == 0
    assert out.splitlines()[0] == HEADER
    rows = rows_of(out)
    marks = {
        definition: Counter((row[-2], row[-1]) for row in rows if row[0] == definition)
        for definition in Definition
    }
    assert marks[Definition.SUBGROUP] == {("True", "False"): 4, ("True", "True"): 2}
    assert marks[Definition.POPULATION] == {}
    assert marks[Definition.FREQUENCY] == {("True", "True"): 2}
    order = list(dict.fromkeys(row[0] for row in rows))
    assert order == [d.value for d in Definition if marks[d]]


def test_the_log_names_the_input_digest_the_join_and_each_definitions_counts(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko")
    code, out, path = run(tmp_path, scenario_a())
    assert code == 0
    found = messages(caplog)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in (
        f"candidates path={path} lines=40 sha256={digest}",
        "request key=<55/female/0 rows=20 lines=20",
        (
            "join key=<55/female/1 lines=20 passed=20 off_cell=0 past_quota=0 joined=20 "
            "newly_held=HP:9000003,MONDO:9000004"
        ),
        (
            "join key=<55/female/0 lines=20 passed=20 off_cell=0 past_quota=0 joined=20 "
            "newly_held=none"
        ),
        "closure definition=per-subgroup before=6 after=2 closed=4 open=2 new=0 share=0.6667",
        (
            "closure definition=population-level before=0 after=0 closed=0 open=0 new=0 "
            "share=undefined"
        ),
    ):
        assert line in found
    report = hashlib.sha256(out.encode()).hexdigest()
    assert found[-1] == f"report rows={len(out.splitlines()) - 1} sha256={report}"


def test_a_second_run_writes_the_same_bytes(tmp_path):
    _, first, _ = run(tmp_path, scenario_a())
    _, second, _ = run(tmp_path, scenario_a())
    assert first == second


def test_a_file_answering_e1s_request_exits_three_and_writes_nothing(tmp_path, caplog):
    e1 = candidate_lines(
        [
            ("<55/female/1", [young(1, cp=1)] * 18),
            (">=55/female/1", [planted(age=60, sex=0, target=1, cp=1)] * 2),
        ]
    )
    code, out, _ = run(tmp_path, e1)
    assert (code, out) == (3, "")
    assert any(
        record.levelno == logging.ERROR
        and "call 2 answers >=55/female/1, a cell the completeness request does not ask for"
        in record.getMessage()
        for record in caplog.records
    )


def test_a_requested_cell_with_too_few_lines_exits_three(tmp_path, caplog):
    short = candidate_lines(
        [("<55/female/0", [young(0, cp=4)] * 20), ("<55/female/1", [young(1, cp=1)] * 19)]
    )
    code, out, _ = run(tmp_path, short)
    assert (code, out) == (3, "")
    assert any(
        record.levelno == logging.ERROR
        and "<55/female/1 has 19 lines for 20 rows" in record.getMessage()
        for record in caplog.records
    )


def errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_key_outside_the_grid_exits_three(tmp_path, caplog):
    stray = candidate_lines(
        [
            ("<55/female/0", [young(0, cp=4)] * 20),
            ("<55/female/1", [young(1, cp=1)] * 20),
            ("<55/female/9", [young(1, cp=1)]),
        ]
    )
    code, out, _ = run(tmp_path, stray)
    assert (code, out) == (3, "")
    assert any("call 3 answers <55/female/9" in message for message in errors(caplog))


def test_a_malformed_file_exits_three(tmp_path, caplog):
    code, out, _ = run(tmp_path, scenario_a().rstrip(b"\n"))
    assert (code, out) == (3, "")
    assert any("has no newline" in message for message in errors(caplog))


def test_a_missing_candidates_file_exits_two_before_any_download(tmp_path):
    def refuse(url: str) -> object:
        pytest.fail(f"fetched {url} for a missing candidates file")

    out = io.StringIO()
    argv = [str(tmp_path / "absent.out"), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=make_context(open_url=refuse), out=out) == 2
    assert out.getvalue() == ""
    assert not (tmp_path / "data").exists()


def test_an_undecoded_line_counts_as_generated_and_joins_nothing(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko")
    items: list[object] = [young(1, cp=1)] * 18 + [Undecoded(Reason.SHORT)] * 2
    data = candidate_lines([("<55/female/0", [young(0, cp=4)] * 20), ("<55/female/1", items)])
    code, _, _ = run(tmp_path, data)
    assert code == 0
    assert any(
        message.startswith(
            "join key=<55/female/1 lines=20 passed=18 off_cell=0 past_quota=0 joined=18 "
        )
        for message in messages(caplog)
    )


def test_an_off_cell_record_is_warned_and_not_joined(tmp_path, caplog):
    items = [young(1, cp=1)] * 19 + [planted(age=60, sex=0, target=1, cp=1)]
    data = candidate_lines([("<55/female/0", [young(0, cp=4)] * 20), ("<55/female/1", items)])
    code, _, _ = run(tmp_path, data)
    assert code == 0
    assert any(
        message.startswith("off_cell key=<55/female/1 dropped=1") for message in warnings_of(caplog)
    )


def test_a_requested_cell_nothing_joins_is_warned(tmp_path, caplog):
    data = candidate_lines(
        [("<55/female/0", [young(0, cp=4, chol=601)] * 20), ("<55/female/1", [young(1)] * 20)]
    )
    code, _, _ = run(tmp_path, data)
    assert code == 0
    found = warnings_of(caplog)
    assert any(message.startswith("unjoined key=<55/female/0") for message in found)
    assert not any(message.startswith("unjoined key=<55/female/1") for message in found)


def held_out_angina(tmp_path: Path) -> bytes:
    """`UCI_X` with angina in the held-out young women without the condition, and no others."""
    training_split(tmp_path / "probe", UCI_X)
    test = split(load_uci(tmp_path / "probe", source=pin(UCI_X))).test
    held = set(test[(test["age"] == 40) & (test["sex"] == 0)].index)
    assert held
    lines: list[str] = []
    for (age, sex, num), size in SIZES.items():
        for _ in range(size):
            record_id = len(lines) + 1
            cp = "1.0" if record_id in held else "4.0" if (age, sex) == (40, 0) else "1.0"
            lines.append(cleveland_line(age=age, sex=sex, num=num, cp=cp))
    data = archive("\n".join(lines) + "\n")
    training_split(tmp_path / "rebuilt", data)
    rebuilt = split(load_uci(tmp_path / "rebuilt", source=pin(data))).test
    assert set(rebuilt[(rebuilt["age"] == 40) & (rebuilt["sex"] == 0)].index) == held
    assert set(rebuilt.loc[sorted(held), "cp"]) == {1}
    return data


def test_the_test_split_never_grounds_the_before_reading(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko")
    data = held_out_angina(tmp_path)
    code, _, _ = run(tmp_path, scenario_a(), make_context(data))
    assert code == 0
    assert any(
        message.startswith("closure definition=per-subgroup before=6 ")
        for message in messages(caplog)
    )


def test_the_before_reading_equals_what_bko_gaps_uci_lists(tmp_path):
    code, out, _ = run(tmp_path, scenario_a())
    assert code == 0
    closure_rows = [
        row[1:9] for row in rows_of(out) if row[0] == Definition.SUBGROUP and row[-2] == "True"
    ]
    pins, served = ladder_releases()
    uci = pin(UCI_X)
    context = gaps_command.Context(
        open_url=RoutingOpener({**served, uci.url: UCI_X}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
    )
    listing = io.StringIO()
    argv = ["uci", "--data-dir", str(tmp_path / "data")]
    assert gaps_command.main(argv, context=context, out=listing) == 0
    gap_rows = [row[1:9] for row in rows_of(listing.getvalue()) if row[0] == Definition.SUBGROUP]
    assert gap_rows
    assert closure_rows == gap_rows


def test_configure_logging_sends_info_and_warnings_to_stderr(monkeypatch):
    """`captureWarnings` is stubbed: the real call is process-wide and outlives the test."""
    recorded: list[dict[str, object]] = []
    captured: list[bool] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(logging, "captureWarnings", captured.append)
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO
    assert captured == [True]


def test_the_module_runs_as_a_program():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    result = subprocess.run(
        [sys.executable, "-m", "bko.fusion", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "candidates" in result.stdout
