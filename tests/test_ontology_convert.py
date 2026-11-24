"""`bko.ontology.convert`: an OWL-only release converted to OBO by ROBOT at fetch time.

Java never runs: `FakeRobot` stands in for ROBOT and writes a canned OBO.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import Final

import pytest

from bko.datasets.errors import UnavailableError
from bko.datasets.fetch import snapshot_dir
from bko.datasets.sources import Source
from bko.ontology.convert import convert_argv, converted
from bko.ontology.elk import IMPORTS
from bko.ontology.errors import ReleaseFormatError
from bko.ontology.releases import load_releases
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import LADDER_MONDO
from tests._ontology_fixtures import FakeRobot, RoutingOpener


JAVA: Final = Path("/opt/jdk/bin/java")
ROBOT: Final = b"robot" * 50
OWL: Final = b'<?xml version="1.0"?>\n<rdf:RDF><owl:Ontology rdf:about="mondo"/></rdf:RDF>\n'


def _java() -> tuple[Path, str]:
    return JAVA, 'openjdk version "21.0.4"'


def owl_pin(body: bytes = OWL) -> Source:
    return dataclasses.replace(pin(body, name="toy-owl"), filename="toy.owl")


def test_the_convert_argv_converts_without_reasoning(tmp_path):
    argv = convert_argv(
        java=JAVA, jar=tmp_path / "robot.jar", owl=tmp_path / "t.owl", out=tmp_path / "t.obo"
    )
    assert argv == [
        str(JAVA),
        "-Xmx4g",
        "-jar",
        str(tmp_path / "robot.jar"),
        "convert",
        "--input",
        str(tmp_path / "t.owl"),
        "--check",
        "false",
        "--format",
        "obo",
        "--output",
        str(tmp_path / "t.obo"),
    ]


@pytest.mark.parametrize("marker", IMPORTS)
def test_an_owl_that_imports_is_refused_before_robot_runs(tmp_path, marker):
    body = OWL + marker + b" <http://example.invalid/other.owl>\n"
    source = owl_pin(body)
    robot = FakeRobot()
    jar = pin(ROBOT, name="robot-jar")
    opener = RoutingOpener({source.url: body})
    with pytest.raises(ReleaseFormatError, match="names an import"):
        converted(source, tmp_path, open_url=opener, resolve_java=_java, robot=jar, runner=robot)
    assert robot.calls == []
    assert [url for url, _ in opener.calls] == [source.url]
    assert not (snapshot_dir(source, tmp_path) / "toy.obo").exists()


def test_a_converted_release_is_reused_and_logs_its_digest(tmp_path, caplog):
    source = owl_pin()
    jar = pin(ROBOT, name="robot-jar")
    robot = FakeRobot(text=LADDER_MONDO)
    opener = RoutingOpener({source.url: OWL, jar.url: ROBOT})
    convert = partial(converted, resolve_java=_java, robot=jar, runner=robot)
    caplog.set_level(logging.INFO, logger="bko.ontology")
    first = load_releases({"MONDO": source}, tmp_path, open_url=opener, convert=convert)
    second = load_releases({"MONDO": source}, tmp_path, open_url=opener, convert=convert)
    assert len(robot.calls) == 1
    assert first["MONDO"].name == second["MONDO"].name == "toy-owl"
    assert first["MONDO"].version == "releases/2026-01-01"
    obo = snapshot_dir(source, tmp_path) / "toy.obo"
    body = LADDER_MONDO.encode()
    digest = hashlib.sha256(body).hexdigest()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("converted ")]
    assert len(lines) == 2
    assert lines[0].startswith("converted release=toy-owl seconds=")
    assert lines[0].endswith(f"path={obo} bytes={len(body)} sha256={digest} reused=false")
    assert (
        lines[1]
        == f"converted release=toy-owl path={obo} bytes={len(body)} sha256={digest} reused=true"
    )
    assert sorted(path.name for path in obo.parent.iterdir() if path.suffix == ".obo") == [
        "toy.obo"
    ]


@pytest.mark.parametrize(
    "failure", [{"status": 1, "stderr": b"boom\n"}, {"timeout": True}, {"write": False}]
)
def test_a_robot_failure_leaves_no_obo(tmp_path, failure):
    robot = FakeRobot(**failure)
    source = owl_pin()
    jar = pin(ROBOT, name="robot-jar")
    opener = RoutingOpener({source.url: OWL, jar.url: ROBOT})
    with pytest.raises(UnavailableError, match="ROBOT"):
        converted(source, tmp_path, open_url=opener, resolve_java=_java, robot=jar, runner=robot)
    assert len(robot.calls) == 1
    kept = sorted(path.name for path in snapshot_dir(source, tmp_path).iterdir())
    assert kept == ["source.json", "toy.obo.stderr", "toy.owl"]
