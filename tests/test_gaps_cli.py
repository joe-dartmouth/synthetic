"""`python -m bko.gaps` over an archive, releases and a population built in tmp_path."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import os
import subprocess
import sys
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from bko.datasets.sources import Source
from bko.datasets.synthea_run import SyntheaRun
from bko.gaps.__main__ import Context, configure_logging, main
from bko.ontology.schema import FieldMapping
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import FakeSynthea, archive, cleveland_line, pin
from tests._gaps_fixtures import (
    ANGINA,
    CELL,
    CONDITION,
    E2_POPULATION,
    ISCHEMIA,
    LADDER_HPO,
    LADDER_MONDO,
    STENOSIS,
    UNSEEN,
)
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

UCI = archive(
    "\n".join(
        cleveland_line(age=63, sex=sex, num=num, cp="4.0")
        for sex in (0, 1)
        for num in (0, 1)
        for _ in range(5)
    )
    + "\n"
)
"""Twenty records aged 63, five per sex and label, none with typical angina (cp 1); one in five of
each cell is held out, so the training split holds four per cell."""

UCI_EXPECTED = (
    "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
    "age_band,sex,target,count\n"
    "bko,hpo,HP:9000002,Chest pain,HP:9000001,Phenotypic abnormality,,,,\n"
    "bko,hpo,HP:9000003,Angina pectoris,HP:9000002,Chest pain,,,,\n"
    "count,,,,,,<55,female,0,0\n"
    "count,,,,,,<55,female,1,0\n"
    "count,,,,,,<55,male,0,0\n"
    "count,,,,,,<55,male,1,0\n"
    "count,,,,,,>=55,female,0,4\n"
    "count,,,,,,>=55,female,1,4\n"
    "count,,,,,,>=55,male,0,4\n"
    "count,,,,,,>=55,male,1,4\n"
)

JAR = b"jar" * 100


def releases() -> tuple[dict[str, Source], dict[str, bytes]]:
    bodies = {"HP": LADDER_HPO.encode(), "MONDO": LADDER_MONDO.encode()}
    pins = {"HP": pin(bodies["HP"], name="hpo"), "MONDO": pin(bodies["MONDO"], name="mondo")}
    return pins, {pins[prefix].url: body for prefix, body in bodies.items()}


def make_context(**change) -> Context:
    pins, served = releases()
    uci = pin(UCI)
    jar = pin(JAR, "synthea-jar")
    context = Context(
        open_url=RoutingOpener({**served, uci.url: UCI, jar.url: JAR}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        code_rows=(ISCHEMIA,),
        synthea_jar=jar,
        resolve_java=lambda: (Path("/opt/jdk/bin/java"), 'openjdk version "21.0.4"'),
        runner=FakeSynthea(tables=E2_POPULATION),
        run=SyntheaRun(seed=1, clinician_seed=1, population=42, snapshot_date=date(2026, 1, 1)),
        ablations={"cell": CELL, "condition": CONDITION},
    )
    return dataclasses.replace(context, **change)


def run(command: str, data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main([command, "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def fetched(context: Context) -> set[str]:
    opener = context.open_url
    assert isinstance(opener, RoutingOpener)
    return {url for url, _ in opener.calls}


def test_uci_lists_both_kinds_of_gap_in_the_training_split(tmp_path, caplog):
    context = make_context()
    with caplog.at_level(logging.INFO, logger="bko"):
        assert run("uci", tmp_path, context) == (0, UCI_EXPECTED)
    assert fetched(context) == {
        context.uci.url,
        context.releases["HP"].url,
        context.releases["MONDO"].url,
    }
    digest = hashlib.sha256(UCI_EXPECTED.encode()).hexdigest()
    for line in (
        "split seed=0 train=16 test=4",
        "scope release=hpo classes=3 statements=2",
        "scope release=mondo classes=3 statements=2",
        "observed dataset=uci-heart-disease records=16 held=1 of=2",
        "fused fragments=3 normalization=fragments",
        "bko dataset=uci-heart-disease ungrounded=2",
        "count dataset=uci-heart-disease flagged=8 below=20",
        f"gaps rows=10 sha256={digest}",
    ):
        assert line in caplog.text


def test_a_second_run_writes_the_same_bytes_and_fetches_nothing(tmp_path):
    assert run("uci", tmp_path, make_context()) == (0, UCI_EXPECTED)
    again = make_context()
    assert run("uci", tmp_path, again) == (0, UCI_EXPECTED)
    assert fetched(again) == set()


def row(concept: str) -> FieldMapping:
    return FieldMapping(when=STENOSIS.when, concept=concept, label="x", source="t", limits="t")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"open_url": RoutingOpener({}, error=urllib.error.URLError("down"))}, "down"),
        ({"field_rows": (row("DOID:1"),)}, "no pinned release for the prefix DOID"),
        ({"field_rows": (row("HP:9999999"),)}, "HP:9999999 is absent"),
    ],
    ids=["unreachable", "unpinned prefix", "absent concept"],
)
def test_a_command_that_cannot_run_exits_two_and_writes_nothing(tmp_path, caplog, change, message):
    with caplog.at_level(logging.ERROR, logger="bko.gaps"):
        assert run("uci", tmp_path, make_context(**change)) == (2, "")
    assert message in caplog.text


def test_a_release_that_contradicts_its_pin_exits_three(tmp_path):
    _, served = releases()
    uci = pin(UCI)
    tampered = {url: b"x" * len(body) for url, body in served.items()}
    context = make_context(open_url=RoutingOpener({**tampered, uci.url: UCI}), uci=uci)
    assert run("uci", tmp_path, context) == (3, "")


def test_a_data_root_that_is_a_file_exits_two(tmp_path):
    root = tmp_path / "file"
    root.write_text("x", encoding="utf-8")
    assert run("uci", root, make_context()) == (2, "")


def test_configure_logging_sends_info_to_stderr(monkeypatch):
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO


def test_the_module_runs_as_a_program():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    result = subprocess.run(
        [sys.executable, "-m", "bko.gaps", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "{uci,synthea}" in result.stdout


ISCHEMIA_ROWS = (
    "mondo,MONDO:9000002,heart disorder,MONDO:9000001,disease,,,",
    "mondo,MONDO:9000003,myocardial ischemia,MONDO:9000002,heart disorder,,,",
)
GRID = (
    ("<60", "female", 1),
    ("<60", "male", 0),
    ("<60", "male", 1),
    (">=60", "female", 0),
    (">=60", "female", 1),
    (">=60", "male", 0),
    (">=60", "male", 1),
)
"""The cells a row lists, in grid order: `<60,female,0` holds twenty records before and after both
ablations and is in no truth, so it is never listed."""
FLAGGED_THROUGHOUT = "True,True,False"


def count_rows(name: str, flags: dict[tuple[str, str, int], str]) -> str:
    lines = []
    for band, sex, target in GRID:
        flag = flags.get((band, sex, target), FLAGGED_THROUGHOUT)
        lines.append(f"{name},count,,,,,,{band},{sex},{target},{flag}\n")
    return "".join(lines)


SYNTHEA_EXPECTED = (
    "ablation,detector,source,antecedent,antecedent_label,consequent,consequent_label,"
    "age_band,sex,target,before,after,truth\n"
    + "".join(f"cell,bko,{row},False,False,True\n" for row in ISCHEMIA_ROWS)
    + count_rows("cell", {(">=60", "male", 1): "False,True,True"})
    + "".join(f"condition,bko,{row},False,True,True\n" for row in ISCHEMIA_ROWS)
    + count_rows(
        "condition",
        {(">=60", "female", 1): "True,True,True", (">=60", "male", 1): "False,True,True"},
    )
)


def test_synthea_reads_each_ablation_with_both_detectors(tmp_path, caplog):
    context = make_context()
    with caplog.at_level(logging.INFO, logger="bko"):
        assert run("synthea", tmp_path, context) == (0, SYNTHEA_EXPECTED)
    assert fetched(context) == {context.synthea_jar.url, context.releases["MONDO"].url}
    for line in (
        "observed dataset=synthea records=42 held=1 of=1 MONDO:9000003=21",
        "ablation name=cell codes=414545008 age_band=>=60 sex=1 edges=60 removed=20",
        "recovery ablation=cell detector=bko truth=2 eligible=2 recovered=0 spurious=0",
        "recovery ablation=cell detector=count truth=1 eligible=1 recovered=1 spurious=0",
        "ablation name=condition codes=414545008 age_band=None sex=None edges=60 removed=21",
        "recovery ablation=condition detector=bko truth=2 eligible=2 recovered=2 spurious=0",
        "recovery ablation=condition detector=count truth=2 eligible=1 recovered=1 spurious=0",
    ):
        assert line in caplog.text


def test_a_second_synthea_run_writes_the_same_bytes_and_fetches_nothing(tmp_path):
    assert run("synthea", tmp_path, make_context()) == (0, SYNTHEA_EXPECTED)
    again = make_context()
    assert run("synthea", tmp_path, again) == (0, SYNTHEA_EXPECTED)
    assert fetched(again) == set()
    assert isinstance(again.runner, FakeSynthea)
    assert again.runner.calls == []


def test_a_statement_ungrounded_throughout_is_listed_before_and_after(tmp_path):
    code, out = run("synthea", tmp_path, make_context(code_rows=(ISCHEMIA, UNSEEN)))
    assert code == 0
    unseen = "mondo,MONDO:9000004,coronary stenosis,MONDO:9000002,heart disorder,,,"
    assert f"cell,bko,{unseen},True,True,False\n" in out
    assert f"condition,bko,{unseen},True,True,False\n" in out


def test_synthea_with_a_concept_its_release_lacks_exits_two(tmp_path, caplog):
    absent = dataclasses.replace(ISCHEMIA, concept="MONDO:9999999")
    with caplog.at_level(logging.ERROR, logger="bko.gaps"):
        assert run("synthea", tmp_path, make_context(code_rows=(absent,))) == (2, "")
    assert "MONDO:9999999 is absent" in caplog.text
