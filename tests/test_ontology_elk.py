"""`python -m bko.ontology elk`: ROBOT's ELK classification beside the hierarchy the loader reads.

Java never runs: a fake runner stands in for ROBOT and writes a canned OBO.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

import bko.ontology.elk as elk_module
from bko.datasets.errors import UnavailableError
from bko.datasets.synthea_run import child_environment
from bko.ontology.__main__ import Context, main
from bko.ontology.elk import (
    ELK_MEMORY_GB,
    Comparison,
    compare_hierarchies,
    read_hierarchy,
    robot_argv,
    run_robot,
)
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import LADDER_HPO, LADDER_MONDO
from tests._ontology_fixtures import RoutingOpener


JAVA: Final = Path("/opt/jdk/bin/java")
VERSION: Final = 'openjdk version "21.0.4"'

ELK_MONDO: Final = LADDER_MONDO.replace(
    "id: MONDO:9000005\nname: unrelated disorder\nis_a: MONDO:9000001\n",
    "id: MONDO:9000005\nname: unrelated disorder\nis_a: MONDO:9000002 ! heart disorder\n",
)
"""ELK's output for the MONDO ladder: the unrelated disorder moved under heart disorder."""

OWL_MONDO: Final = b'<?xml version="1.0"?>\n<rdf:RDF><owl:Ontology rdf:about="mondo"/></rdf:RDF>\n'
OWL_HPO: Final = b'<?xml version="1.0"?>\n<rdf:RDF><owl:Ontology rdf:about="hp"/></rdf:RDF>\n'
ROBOT: Final = b"robot" * 50


def links(**parents: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return {
        child.replace("_", ":"): tuple(p.replace("_", ":") for p in ps)
        for child, ps in parents.items()
    }


LADDER: Final = links(A_1=["A_2"], A_2=["A_3"], A_3=[], B_1=["A_3"])


def test_identical_hierarchies_differ_nowhere():
    found = compare_hierarchies(LADDER, LADDER, {"A:1"})
    assert isinstance(found, Comparison)
    assert found.differing == 0
    assert found.concepts == 3


def test_a_link_only_elk_holds_is_reported():
    elk = {**LADDER, "A:1": ("A:2", "B:1")}
    found = compare_hierarchies(LADDER, {**elk, "B:1": ("A:3",)}, {"A:1"})
    assert found.only_elk == (("A:1", "B:1"),)
    assert found.only_loaded == ()
    assert found.differing == 1


def test_a_link_only_the_loaded_release_holds_is_reported():
    found = compare_hierarchies(LADDER, {**LADDER, "A:2": ()}, {"A:1"})
    assert found.only_loaded == (("A:1", "A:3"), ("A:2", "A:3"))
    assert found.only_elk == ()


def test_a_link_outside_the_scope_is_ignored():
    found = compare_hierarchies(LADDER, {**LADDER, "B:1": ("A:2",)}, {"A:1"})
    assert found.differing == 0


def test_a_redundant_direct_edge_is_not_a_difference():
    planted = {**LADDER, "A:1": ("A:2", "A:3")}
    assert compare_hierarchies(planted, LADDER, {"A:1"}).differing == 0


def test_an_equivalence_cycle_terminates():
    cycle = {**LADDER, "A:2": ("A:3", "A:1")}
    found = compare_hierarchies(LADDER, cycle, {"A:1"})
    assert found.only_elk == (("A:2", "A:1"),)


def test_the_obo_reader_keeps_each_terms_is_a_targets_only(tmp_path):
    path = tmp_path / "out.obo"
    path.write_text(
        'format-version: 1.2\n\n[Term]\nid: A:1\nis_a: A:2 {source="x"} ! two\n'
        "relationship: part_of A:9\n\n[Typedef]\nid: part_of\nis_a: A:8\n",
        encoding="utf-8",
    )
    assert read_hierarchy(path) == {"A:1": ("A:2",)}


def test_the_argv_runs_robots_elk_then_converts_to_obo(tmp_path):
    argv = robot_argv(
        java=JAVA, jar=tmp_path / "robot.jar", owl=tmp_path / "m.owl", out=tmp_path / "m.obo"
    )
    assert ELK_MEMORY_GB == 16
    assert argv == [
        str(JAVA),
        "-Xmx16g",
        "-jar",
        str(tmp_path / "robot.jar"),
        "reason",
        "--reasoner",
        "ELK",
        "--equivalent-classes-allowed",
        "all",
        "--input",
        str(tmp_path / "m.owl"),
        "convert",
        "--check",
        "false",
        "--format",
        "obo",
        "--output",
        str(tmp_path / "m.obo"),
    ]


class FakeRobot:
    """Writes a canned OBO to the argv's output, or fails as told."""

    def __init__(self, status: int = 0, stderr: bytes = b"", timeout: bool = False) -> None:
        self.status = status
        self.stderr = stderr
        self.timeout = timeout
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, bytes]:
        self.calls.append(list(argv))
        if self.timeout:
            raise subprocess.TimeoutExpired(list(argv), timeout)
        out = Path(argv[argv.index("--output") + 1])
        owl = Path(argv[argv.index("--input") + 1]).read_bytes()
        if self.status == 0:
            out.write_text(ELK_MONDO if b"mondo" in owl else LADDER_HPO, encoding="utf-8")
        return self.status, self.stderr


def elk_context(robot: FakeRobot, *, mondo_owl: bytes = OWL_MONDO) -> Context:
    obo = {"HP": LADDER_HPO.encode(), "MONDO": LADDER_MONDO.encode()}
    releases = {"HP": pin(obo["HP"], name="hpo"), "MONDO": pin(obo["MONDO"], name="mondo")}
    owl = {"HP": pin(OWL_HPO, name="hpo-owl"), "MONDO": pin(mondo_owl, name="mondo-owl")}
    jar = pin(ROBOT, name="robot-jar")
    bodies = {
        **{releases[p].url: obo[p] for p in obo},
        owl["HP"].url: OWL_HPO,
        owl["MONDO"].url: mondo_owl,
        jar.url: ROBOT,
    }
    return Context(
        open_url=RoutingOpener(bodies),
        releases=releases,
        owl_releases=owl,
        robot=jar,
        resolve_java=lambda: (JAVA, VERSION),
        robot_runner=robot,
        concepts=frozenset({"MONDO:9000003", "MONDO:9000005", "HP:9000003"}),
    )


def run_elk(tmp_path: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["elk", "--data-dir", str(tmp_path / "data")], context=context, out=out)
    return code, out.getvalue()


def test_elk_lists_each_link_one_side_holds_and_logs_its_identity(tmp_path, caplog):
    robot = FakeRobot()
    caplog.set_level(logging.INFO)
    code, out = run_elk(tmp_path, elk_context(robot))
    assert code == 0
    assert out == ("release,child,ancestor,side\nmondo,MONDO:9000005,MONDO:9000002,elk\n")
    assert len(robot.calls) == 2
    assert all(call[:2] == [str(JAVA), "-Xmx16g"] for call in robot.calls)
    sha = hashlib.sha256(ROBOT).hexdigest()
    assert f"elk tool robot_sha256={sha} java={VERSION}" in caplog.messages
    assert any(m.startswith("elk classified release=mondo seconds=") for m in caplog.messages)
    assert "elk compare release=hpo concepts=3 only_loaded=0 only_elk=0" in caplog.messages
    assert "elk compare release=mondo concepts=5 only_loaded=0 only_elk=1" in caplog.messages
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert f"elk differing links=1 sha256={digest}" in caplog.messages
    rows = list(csv.DictReader(io.StringIO(out)))
    assert {row["side"] for row in rows} == {"elk"}


def test_an_owl_that_imports_is_refused_before_java_runs(tmp_path, caplog):
    robot = FakeRobot()
    importing = OWL_MONDO.replace(b"</rdf:RDF>", b'<owl:imports rdf:resource="x"/></rdf:RDF>')
    code, out = run_elk(tmp_path, elk_context(robot, mondo_owl=importing))
    assert (code, out) == (3, "")
    assert robot.calls == []
    assert "owl:imports" in caplog.text


def test_a_non_zero_exit_keeps_robots_stderr_byte_for_byte(tmp_path, caplog):
    stderr = b"ERROR \xff\x00 heap\r\nsecond line"
    code, out = run_elk(tmp_path, elk_context(FakeRobot(status=1, stderr=stderr)))
    assert (code, out) == (2, "")
    kept = list((tmp_path / "data" / "elk").glob("*.stderr"))
    assert len(kept) == 1
    assert kept[0].read_bytes() == stderr
    assert "ROBOT exited 1" in caplog.text
    assert str(kept[0]) in caplog.text


def test_a_timeout_is_a_refusal(tmp_path, caplog):
    code, out = run_elk(tmp_path, elk_context(FakeRobot(timeout=True)))
    assert (code, out) == (2, "")
    assert "ROBOT did not finish in" in caplog.text


def test_classify_raises_unavailable_on_a_timeout(tmp_path):
    with pytest.raises(UnavailableError, match="did not finish"):
        elk_module.classify(
            tmp_path / "m.owl",
            tmp_path / "m.obo",
            java=JAVA,
            jar=tmp_path / "robot.jar",
            runner=FakeRobot(timeout=True),
        )


def test_the_child_environment_comes_from_synthea_run(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=None, stderr=b"warn")

    monkeypatch.setattr(elk_module.subprocess, "run", fake_run)
    assert run_robot(["java", "-version"], 5.0) == (0, b"warn")
    assert seen["env"] == child_environment()
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["timeout"] == pytest.approx(5.0)
