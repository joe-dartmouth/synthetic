"""`python -m bko.ontology elk`: ROBOT's ELK classification beside the hierarchy the loader reads.

Java never runs: a fake runner stands in for ROBOT and writes a canned OBO.
"""

from __future__ import annotations

import csv
import dataclasses
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
    minimal,
    read_hierarchy,
    refuse_imports,
    robot_argv,
    run_robot,
)
from bko.ontology.errors import ReleaseFormatError
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import LADDER_HPO, LADDER_MONDO
from tests._ontology_fixtures import RoutingOpener


JAVA: Final = Path("/opt/jdk/bin/java")
VERSION: Final = 'openjdk version "21.0.4"'

ELK_MONDO: Final = LADDER_MONDO.replace(
    "id: MONDO:9000005\nname: unrelated disorder\nis_a: MONDO:9000001\n",
    "id: MONDO:9000005\nname: unrelated disorder\nis_a: MONDO:9000002 ! heart disorder\n",
).replace(
    "id: MONDO:9000003\nname: myocardial ischemia\nis_a: MONDO:9000002\n",
    "id: MONDO:9000003\nname: myocardial ischemia\nis_a: MONDO:9000001\n",
)
"""ELK's output for the MONDO ladder: the unrelated disorder moved under heart disorder, and
myocardial ischemia moved from under heart disorder to under disease."""

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
    assert (found.minimal_loaded, found.minimal_elk) == (0, 0)
    assert (found.absent_loaded, found.absent_elk) == (0, 0)


def test_an_ancestor_elk_adds_joins_the_scope_with_its_own_pairs():
    elk = {**LADDER, "A:1": ("A:2", "B:1"), "B:1": ("A:3", "C:1"), "C:1": ()}
    found = compare_hierarchies(LADDER, elk, {"A:1"})
    assert found.concepts == 5
    assert found.only_elk == (("A:1", "B:1"), ("A:1", "C:1"), ("B:1", "C:1"))
    assert found.only_loaded == ()
    assert found.minimal_elk == 2
    assert (found.absent_loaded, found.absent_elk) == (1, 0)


def test_a_pair_implied_by_another_is_not_minimal():
    pairs = [("A:1", "A:3"), ("A:2", "A:3")]
    assert minimal(LADDER, pairs) == 1
    assert minimal(LADDER, pairs[:1]) == 1


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
    assert found.minimal_elk == 1


def test_mutually_implied_pairs_both_count_as_minimal():
    cycle = links(A_1=["A_2"], A_2=["A_1"])
    assert minimal(cycle, [("A:1", "A:2"), ("A:2", "A:1")]) == 2


def test_the_obo_reader_keeps_each_terms_is_a_targets_only(tmp_path):
    path = tmp_path / "out.obo"
    path.write_text(
        'format-version: 1.2\n\n[Term]\nid: A:1\nis_a: A:2 {source="x"} ! two\n'
        "relationship: part_of A:9\n\n[Typedef]\nid: part_of\nis_a: A:8\n",
        encoding="utf-8",
    )
    assert read_hierarchy(path) == {"A:1": ("A:2",)}


def test_the_obo_reader_sorts_and_dedupes_reads_equivalence_both_ways_and_keeps_obsolete(
    tmp_path,
):
    path = tmp_path / "out.obo"
    path.write_text(
        "[Term]\nid: A:1\nis_a: A:3\nis_a: A:2\nis_a: A:3\nequivalent_to: B:1 ! b\n\n"
        "[Term]\nid: A:9\nis_obsolete: true\nis_a: A:8\n",
        encoding="utf-8",
    )
    assert read_hierarchy(path) == {"A:1": ("A:2", "A:3", "B:1"), "B:1": ("A:1",), "A:9": ("A:8",)}


@pytest.mark.parametrize("line", ["id:", "is_a:  ", "equivalent_to:"])
def test_the_obo_reader_refuses_a_line_with_no_value(tmp_path, line):
    path = tmp_path / "out.obo"
    path.write_text(f"[Term]\nid: A:1\n{line}\n".replace("id: A:1\nid:", "id:"), encoding="utf-8")
    with pytest.raises(ReleaseFormatError, match="has no value"):
        read_hierarchy(path)


@pytest.mark.parametrize(
    "marker", [b"owl:imports", b"owl#imports", b"Import(<http://x>)", b"<Import>x</Import>"]
)
def test_refuse_imports_finds_each_spelling_across_a_chunk_boundary(monkeypatch, tmp_path, marker):
    monkeypatch.setattr(elk_module, "_CHUNK", 8)
    path = tmp_path / "m.owl"
    for offset in range(8):
        path.write_bytes(b"x" * (13 + offset) + marker + b"y" * 20)
        with pytest.raises(ReleaseFormatError, match="names an import"):
            refuse_imports(path)
    path.write_bytes(b"x" * 40)
    refuse_imports(path)


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
    """Writes its stderr, then a canned OBO to the argv's output, or fails as told."""

    def __init__(
        self, status: int = 0, stderr: bytes = b"", timeout: bool = False, write: bool = True
    ) -> None:
        self.status = status
        self.stderr = stderr
        self.timeout = timeout
        self.write = write
        self.calls: list[list[str]] = []
        self.cwds: list[Path] = []

    def __call__(self, argv: Sequence[str], timeout: float, cwd: Path, stderr: Path) -> int:
        self.calls.append(list(argv))
        self.cwds.append(cwd)
        stderr.write_bytes(self.stderr)
        if self.timeout:
            raise subprocess.TimeoutExpired(list(argv), timeout)
        if self.status == 0 and self.write:
            out = Path(argv[argv.index("--output") + 1])
            owl = Path(argv[argv.index("--input") + 1]).read_bytes()
            out.write_text(ELK_MONDO if b"mondo" in owl else LADDER_HPO, encoding="utf-8")
        return self.status


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
    assert out == (
        "release,child,ancestor,side\n"
        "mondo,MONDO:9000003,MONDO:9000002,loaded\n"
        "mondo,MONDO:9000005,MONDO:9000002,elk\n"
    )
    assert len(robot.calls) == 2
    assert all(call[:2] == [str(JAVA), "-Xmx16g"] for call in robot.calls)
    owls = [Path(call[call.index("--input") + 1]) for call in robot.calls]
    assert robot.cwds == [owl.parent for owl in owls]
    context = elk_context(robot)
    sha = hashlib.sha256(ROBOT).hexdigest()
    assert (
        f"elk tool robot_url={context.robot.url} robot_sha256={sha} java={VERSION} "
        "xmx_gb=16 timeout_s=7200"
    ) in caplog.messages
    mondo_sha = context.owl_releases["MONDO"].sha256
    assert (
        f"elk classify release=mondo owl_sha256={mondo_sha} xmx_gb=16 timeout_s=7200"
    ) in caplog.messages
    classified = next((tmp_path / "data" / "elk").glob("mondo-owl-*.obo"))
    body = classified.read_bytes()
    assert any(
        m.startswith("elk classified release=mondo seconds=")
        and m.endswith(
            f" path={classified} bytes={len(body)} sha256={hashlib.sha256(body).hexdigest()}"
        )
        for m in caplog.messages
    )
    assert (
        "elk compare release=hpo loaded_terms=3 elk_classes=3 concepts=3 absent_loaded=0 "
        "absent_elk=0 only_loaded=0 only_elk=0 minimal_loaded=0 minimal_elk=0"
    ) in caplog.messages
    assert (
        "elk compare release=mondo loaded_terms=6 elk_classes=6 concepts=5 absent_loaded=0 "
        "absent_elk=0 only_loaded=1 only_elk=1 minimal_loaded=1 minimal_elk=1"
    ) in caplog.messages
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert f"elk differing ancestor_pairs=2 sha256={digest}" in caplog.messages
    rows = list(csv.DictReader(io.StringIO(out)))
    assert [row["side"] for row in rows] == ["loaded", "elk"]


def test_java_is_resolved_before_any_download(tmp_path):
    def missing() -> tuple[Path, str]:
        raise UnavailableError("no JDK")

    context = dataclasses.replace(elk_context(FakeRobot()), resolve_java=missing)
    code, out = run_elk(tmp_path, context)
    assert (code, out) == (2, "")
    assert not (tmp_path / "data").exists()


def test_an_owl_that_imports_is_refused_before_java_runs(tmp_path, caplog):
    robot = FakeRobot()
    importing = OWL_MONDO.replace(b"</rdf:RDF>", b'<owl:imports rdf:resource="x"/></rdf:RDF>')
    code, out = run_elk(tmp_path, elk_context(robot, mondo_owl=importing))
    assert (code, out) == (3, "")
    assert robot.calls == []
    assert "names an import" in caplog.text


def test_a_non_zero_exit_keeps_robots_stderr_byte_for_byte(tmp_path, caplog):
    stderr = b"ERROR \xff\x00 heap\r\nsecond line"
    code, out = run_elk(tmp_path, elk_context(FakeRobot(status=1, stderr=stderr)))
    assert (code, out) == (2, "")
    kept = list((tmp_path / "data" / "elk").glob("*.stderr"))
    assert len(kept) == 1
    assert kept[0].read_bytes() == stderr
    assert "ROBOT exited 1 after " in caplog.text
    assert str(kept[0]) in caplog.text
    assert "ending: ERROR" in caplog.text


def test_a_timeout_keeps_robots_partial_stderr_and_names_it(tmp_path, caplog):
    stderr = b"".join(b"line %d\n" % n for n in range(9))
    code, out = run_elk(tmp_path, elk_context(FakeRobot(timeout=True, stderr=stderr)))
    assert (code, out) == (2, "")
    (kept,) = (tmp_path / "data" / "elk").glob("*.stderr")
    assert kept.read_bytes() == stderr
    assert "ROBOT did not finish in" in caplog.text
    assert f"its stderr is at {kept}, ending: line 4 | line 5 | line 6 | line 7 | line 8" in (
        caplog.text
    )


def test_an_exit_0_that_writes_nothing_is_a_refusal_and_stale_files_go_first(tmp_path):
    out, kept = tmp_path / "m.obo", tmp_path / "m.obo.stderr"
    out.write_text("stale", encoding="utf-8")
    kept.write_bytes(b"stale")
    with pytest.raises(UnavailableError, match=r"ROBOT exited 0 after .* but wrote no"):
        elk_module.classify(
            tmp_path / "m.owl",
            out,
            java=JAVA,
            jar=tmp_path / "robot.jar",
            runner=FakeRobot(write=False),
        )
    assert not out.exists()
    assert kept.read_bytes() == b""


def test_classify_raises_unavailable_on_a_timeout(tmp_path):
    with pytest.raises(UnavailableError, match="did not finish"):
        elk_module.classify(
            tmp_path / "m.owl",
            tmp_path / "m.obo",
            java=JAVA,
            jar=tmp_path / "robot.jar",
            runner=FakeRobot(timeout=True),
        )


def test_the_real_runner_writes_stderr_to_the_file_in_the_given_directory(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        kwargs["stderr"].write(b"warn")
        return subprocess.CompletedProcess(argv, 3, stdout=None, stderr=None)

    monkeypatch.setattr(elk_module.subprocess, "run", fake_run)
    kept = tmp_path / "robot.stderr"
    assert run_robot(["java", "-version"], 5.0, tmp_path, kept) == 3
    assert kept.read_bytes() == b"warn"
    assert seen["cwd"] == tmp_path
    assert seen["env"] == child_environment()
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["timeout"] == pytest.approx(5.0)
