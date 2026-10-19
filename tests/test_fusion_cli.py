"""`python -m bko.fusion` over an archive, releases and a candidates file built in tmp_path."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import logging
from collections import Counter
from pathlib import Path

import pytest

from bko.datasets.sources import Source
from bko.fusion.__main__ import Context, main
from bko.gaps.definitions import Definition
from tests._datasets_fixtures import pin
from tests._fusion_fixtures import UCI_X, candidate_lines, scenario_a, young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
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
