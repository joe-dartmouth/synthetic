"""`python -m bko.gaps` over an archive, releases and a population built in tmp_path."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import io
import logging
import os
import subprocess
import sys
import urllib.error
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

import bko.gaps.experiment
from bko.datasets.synthea_run import SyntheaRun
from bko.gaps.__main__ import Context, configure_logging, main
from bko.gaps.experiment import Recovery
from bko.ontology.schema import FieldMapping
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import FakeSynthea, archive, cleveland_line, pin
from tests._gaps_fixtures import (
    ANGINA,
    CELL,
    CONDITION,
    E2_POPULATION,
    ISCHEMIA,
    STENOSIS,
    UNSEEN,
    ladder_releases,
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

H21 = "hpo,HP:9000002,Chest pain,HP:9000001,Phenotypic abnormality"
H32 = "hpo,HP:9000003,Angina pectoris,HP:9000002,Chest pain"
M21 = "mondo,MONDO:9000002,heart disorder,MONDO:9000001,disease"
M42 = "mondo,MONDO:9000004,coronary stenosis,MONDO:9000002,heart disorder"
UCI_CELLS = tuple(
    f"{band},{sex},{target}"
    for band in ("<55", ">=55")
    for sex in ("female", "male")
    for target in (0, 1)
)
"""The eight cells in grid order; the four `<55` cells hold no training record."""
EMPTY_WITH_DISEASE = frozenset({"<55,female,1", "<55,male,1"})
"""Empty cells with disease assume nothing, so every statement is their per-subgroup gap; an empty
cell without disease assumes the stenosis only disease rows name, which grounds MONDO's chain."""


def uci_rows(detector: str, statements: Callable[[str], tuple[str, ...]]) -> str:
    return "".join(
        f"{detector},{statement},{cell},\n" for cell in UCI_CELLS for statement in statements(cell)
    )


UCI_EXPECTED = (
    "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
    "age_band,sex,target,count\n"
    + uci_rows(
        "per-subgroup",
        lambda cell: (H21, H32, M21, M42) if cell in EMPTY_WITH_DISEASE else (H21, H32),
    )
    + uci_rows("population-level", lambda _: (H21, H32))
    + "count,,,,,,<55,female,0,0\n"
    "count,,,,,,<55,female,1,0\n"
    "count,,,,,,<55,male,0,0\n"
    "count,,,,,,<55,male,1,0\n"
    "count,,,,,,>=55,female,0,4\n"
    "count,,,,,,>=55,female,1,4\n"
    "count,,,,,,>=55,male,0,4\n"
    "count,,,,,,>=55,male,1,4\n"
)

JAR = b"jar" * 100


def make_context(**change) -> Context:
    pins, served = ladder_releases()
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


def test_uci_lists_each_definitions_gaps_and_the_count_in_the_training_split(tmp_path, caplog):
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
        f"scipy={importlib.metadata.version('scipy')}",
        "split seed=0 train=16 test=4",
        "scope release=hpo classes=3 statements=2",
        "scope release=mondo classes=3 statements=2",
        "observed dataset=uci-heart-disease records=16 held=1 of=2",
        (
            "cell dataset=uci-heart-disease key=<55/female/0 records=0 held=0 of=2 "
            "assumed=MONDO:9000004\n"
        ),
        "cell dataset=uci-heart-disease key=>=55/male/1 records=4 held=1 of=2 assumed=\n",
        "fused fragments=3 normalization=fragments",
        "name=uci-heart-disease\n",
        "bko definition=per-subgroup dataset=uci-heart-disease gaps=20 cells=8",
        "bko definition=population-level dataset=uci-heart-disease gaps=16 cells=8",
        "bko definition=frequency-aware dataset=uci-heart-disease gaps=0 cells=0",
        "count dataset=uci-heart-disease flagged=8 below=20",
        f"gaps rows=44 sha256={digest}",
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
    _, served = ladder_releases()
    uci = pin(UCI)
    tampered = {url: b"x" * len(body) for url, body in served.items()}
    context = make_context(open_url=RoutingOpener({**tampered, uci.url: UCI}), uci=uci)
    assert run("uci", tmp_path, context) == (3, "")


def test_a_data_root_that_is_a_file_exits_two(tmp_path):
    root = tmp_path / "file"
    root.write_text("x", encoding="utf-8")
    assert run("uci", root, make_context()) == (2, "")


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


ISCHEMIA_STATEMENTS = (
    "mondo,MONDO:9000002,heart disorder,MONDO:9000001,disease",
    "mondo,MONDO:9000003,myocardial ischemia,MONDO:9000002,heart disorder",
)
BKO_GRID = (("<60", "female"), ("<60", "male"), (">=60", "female"), (">=60", "male"))
"""The definitions' age-sex cells in grid order; they carry no label."""


def bko_rows(name: str, detector: str, marks: dict[tuple[str, str], str]) -> str:
    """The ischemia statements of each cell `marks` lists, in grid order, with its flags."""
    return "".join(
        f"{name},{detector},{statement},{band},{sex},,{marks[band, sex]}\n"
        for band, sex in BKO_GRID
        if (band, sex) in marks
        for statement in ISCHEMIA_STATEMENTS
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
    + bko_rows(
        "cell",
        "per-subgroup",
        {
            ("<60", "female"): "True,True,False",
            ("<60", "male"): "True,True,False",
            (">=60", "male"): "False,True,True",
        },
    )
    + bko_rows("cell", "population-level", {(">=60", "male"): "False,False,True"})
    + bko_rows(
        "cell",
        "frequency-aware",
        {("<60", "female"): "True,True,False", (">=60", "male"): "False,False,True"},
    )
    + count_rows("cell", {(">=60", "male", 1): "False,True,True"})
    + bko_rows(
        "condition",
        "per-subgroup",
        {
            ("<60", "female"): "True,True,False",
            ("<60", "male"): "True,True,False",
            (">=60", "female"): "False,True,True",
            (">=60", "male"): "False,True,True",
        },
    )
    + bko_rows(
        "condition",
        "population-level",
        {
            ("<60", "female"): "False,True,False",
            ("<60", "male"): "False,True,False",
            (">=60", "female"): "False,True,True",
            (">=60", "male"): "False,True,True",
        },
    )
    + bko_rows(
        "condition",
        "frequency-aware",
        {
            ("<60", "female"): "True,False,False",
            (">=60", "female"): "False,False,True",
            (">=60", "male"): "False,False,True",
        },
    )
    + count_rows(
        "condition",
        {(">=60", "female", 1): "True,True,True", (">=60", "male", 1): "False,True,True"},
    )
)


def test_synthea_reads_each_ablation_with_every_detector(tmp_path, caplog):
    context = make_context()
    with caplog.at_level(logging.INFO, logger="bko"):
        assert run("synthea", tmp_path, context) == (0, SYNTHEA_EXPECTED)
    assert fetched(context) == {context.synthea_jar.url, context.releases["MONDO"].url}
    for line in (
        "observed dataset=synthea records=42 held=1 of=1 MONDO:9000003=21",
        (
            "ablation name=cell codes=414545008 age_band=>=60 sex=1 edges=60 fraction=1.0 "
            "seed=0 removed=20"
        ),
        "cell ablation=cell when=after key=>=60/male records=0 held=0 of=1 assumed=\n",
        "bko ablation=cell definition=per-subgroup before=4 after=6 listed=6",
        "bko ablation=cell definition=population-level before=0 after=0 listed=2",
        "bko ablation=cell definition=frequency-aware before=2 after=2 listed=4",
        (
            "flagged ablation=cell when=before key=<60/female concept=MONDO:9000003 shown=0 "
            "records=20 reference_shown=21 reference_records=22 mass=1.0000"
        ),
        (
            "flagged ablation=cell when=after key=<60/female concept=MONDO:9000003 shown=0 "
            "records=20 reference_shown=1 reference_records=2 mass=1.0000"
        ),
        (
            "recovery ablation=cell detector=per-subgroup truth=2 eligible=2 recovered=2 "
            "outside=4 new_outside=0 vanished=0"
        ),
        (
            "recovery ablation=cell detector=population-level truth=2 eligible=2 recovered=0 "
            "outside=0 new_outside=0 vanished=0"
        ),
        (
            "recovery ablation=cell detector=frequency-aware truth=2 eligible=2 recovered=0 "
            "outside=2 new_outside=0 vanished=0"
        ),
        (
            "recovery ablation=cell detector=count truth=1 eligible=1 recovered=1 "
            "outside=6 new_outside=0 vanished=0"
        ),
        (
            "ablation name=condition codes=414545008 age_band=None sex=None edges=60 "
            "fraction=1.0 seed=0 removed=21"
        ),
        (
            "recovery ablation=condition detector=per-subgroup truth=4 eligible=4 recovered=4 "
            "outside=4 new_outside=0 vanished=0"
        ),
        (
            "recovery ablation=condition detector=population-level truth=4 eligible=4 "
            "recovered=4 outside=4 new_outside=4 vanished=0"
        ),
        (
            "recovery ablation=condition detector=frequency-aware truth=4 eligible=4 "
            "recovered=0 outside=0 new_outside=0 vanished=2"
        ),
        (
            "recovery ablation=condition detector=count truth=2 eligible=1 recovered=1 "
            "outside=5 new_outside=0 vanished=0"
        ),
        (
            "detection ablation=condition detector=per-subgroup sensitivity=1.0000 "
            "specificity=1.0000"
        ),
        (
            "detection ablation=condition detector=population-level sensitivity=1.0000 "
            "specificity=0.0000"
        ),
        ("detection ablation=condition detector=count sensitivity=1.0000 specificity=1.0000"),
    ):
        assert line in caplog.text
    assert "invariant" not in caplog.text


def test_a_closed_detector_reading_new_gaps_outside_the_truth_is_warned(
    tmp_path, caplog, monkeypatch
):
    broken = Recovery(truth=1, eligible=1, recovered=1, outside=1, new_outside=1, vanished=0)
    monkeypatch.setattr(bko.gaps.experiment, "recovery", lambda *_: broken)
    with caplog.at_level(logging.WARNING, logger="bko.gaps"):
        assert run("synthea", tmp_path, make_context())[0] == 0
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [message.split(":")[0] for message in warned] == [
        f"invariant ablation={name} detector={detector}"
        for name in ("cell", "condition")
        for detector in ("per-subgroup", "count")
    ]


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
    unseen = "mondo,MONDO:9000004,coronary stenosis,MONDO:9000002,heart disorder"
    for ablation in ("cell", "condition"):
        for detector in ("per-subgroup", "population-level"):
            for band, sex in BKO_GRID:
                assert f"{ablation},{detector},{unseen},{band},{sex},,True,True,False\n" in out
        assert f"{ablation},frequency-aware,{unseen}," not in out


def test_synthea_with_a_concept_its_release_lacks_exits_two(tmp_path, caplog):
    absent = dataclasses.replace(ISCHEMIA, concept="MONDO:9999999")
    with caplog.at_level(logging.ERROR, logger="bko.gaps"):
        assert run("synthea", tmp_path, make_context(code_rows=(absent,))) == (2, "")
    assert "MONDO:9999999 is absent" in caplog.text
