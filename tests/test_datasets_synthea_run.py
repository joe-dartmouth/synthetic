"""Generating a pinned Synthea population: argv, Java, atomic directories, reuse."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from bko.datasets.errors import ContractError, UnavailableError
from bko.datasets.synthea_run import (
    CONFIG,
    RECORD_NAME,
    SyntheaRun,
    ensure_population,
    find_java,
    population_dir,
    probe_version,
    run_java,
    synthea_argv,
)
from tests._datasets_fixtures import FakeOpener, FakeSynthea, pin


RUN = SyntheaRun(seed=7, clinician_seed=8, population=10, snapshot_date=date(2026, 1, 1))
JAR = b"not really a jar\n" * 64
JAVA = (Path("/opt/jdk/bin/java"), 'openjdk version "21.0.4" 2024-07-16')


def generate(tmp_path, runner=None, opener=None):
    return ensure_population(
        RUN,
        tmp_path,
        jar_source=pin(JAR, "synthea-jar"),
        open_url=opener or FakeOpener(JAR),
        resolve_java=lambda: JAVA,
        runner=runner or FakeSynthea(),
    )


def test_argv_pins_every_output_changing_parameter(tmp_path):
    argv = synthea_argv(RUN, java=Path("/j"), jar=Path("/s.jar"), out_dir=tmp_path)
    assert argv[:4] == ["/j", "-Duser.timezone=UTC", "-jar", "/s.jar"]
    pairs = dict(zip(argv[4:14:2], argv[5:14:2], strict=True))
    assert pairs == {"-s": "7", "-cs": "8", "-r": "20260101", "-e": "20260101", "-p": "10"}
    assert f"--exporter.baseDirectory={tmp_path.resolve()}/" in argv
    assert set(CONFIG) <= set(argv)
    assert "--generate.thread_pool_size=1" in CONFIG
    assert "--exporter.years_of_history=0" in CONFIG
    assert argv[-1] == "Massachusetts"


def test_a_new_population_is_generated_with_its_record(tmp_path):
    runner = FakeSynthea()
    directory = generate(tmp_path, runner)
    assert directory == population_dir(RUN, tmp_path, pin(JAR, "synthea-jar"))
    record = json.loads((directory / RECORD_NAME).read_text(encoding="utf-8"))
    assert record["run"] == {
        "seed": 7,
        "clinician_seed": 8,
        "population": 10,
        "snapshot_date": "2026-01-01",
        "state": "Massachusetts",
    }
    assert record["java_version"] == JAVA[1]
    assert record["jar_sha256"] == pin(JAR, "synthea-jar").sha256
    assert set(record["tables"]) == {"patients.csv", "conditions.csv"}
    assert record["tables"]["patients.csv"]["rows"] == 5
    assert len(runner.calls) == 1


def test_a_present_population_is_reused(tmp_path):
    runner = FakeSynthea()
    opener = FakeOpener(JAR)
    first = generate(tmp_path, runner, opener)
    second = generate(tmp_path, runner, opener)
    assert first == second
    assert len(runner.calls) == 1
    assert len(opener.calls) == 1


def test_a_population_whose_csv_changed_is_regenerated(tmp_path, caplog):
    runner = FakeSynthea()
    directory = generate(tmp_path, runner)
    (directory / "csv" / "patients.csv").write_text("Id\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        generate(tmp_path, runner)
    assert len(runner.calls) == 2
    assert "fails its record" in caplog.text


def test_a_failed_run_leaves_no_population_and_keeps_its_log(tmp_path):
    with pytest.raises(UnavailableError, match="Synthea exited 3"):
        generate(tmp_path, FakeSynthea(status=3))
    directory = population_dir(RUN, tmp_path, pin(JAR, "synthea-jar"))
    assert not directory.exists()
    assert (
        directory.with_name(directory.name + ".log").read_text(encoding="utf-8") == "fake synthea\n"
    )


def test_a_run_that_writes_no_csv_is_refused(tmp_path):
    with pytest.raises(UnavailableError, match=r"wrote no patients\.csv"):
        generate(tmp_path, FakeSynthea(write=False))


def test_a_timeout_is_unavailable(tmp_path):
    def slow(argv, cwd):
        raise subprocess.TimeoutExpired(argv, 1.0)

    with pytest.raises(UnavailableError, match="did not finish"):
        generate(tmp_path, slow)


def test_a_jar_that_fails_its_pin_is_never_run(tmp_path):
    runner = FakeSynthea()
    with pytest.raises(ContractError, match="pinned"):
        generate(tmp_path, runner, FakeOpener(b"x" * len(JAR)))
    assert runner.calls == []


def test_probe_version_reads_stderr(tmp_path):
    script = tmp_path / "java"
    script.write_text("#!/bin/sh\necho 'openjdk version \"21.0.4\"' >&2\n", encoding="utf-8")
    script.chmod(0o755)
    assert probe_version(script) == 'openjdk version "21.0.4"'


def test_probe_version_refuses_a_failing_java(tmp_path):
    script = tmp_path / "java"
    script.write_text("#!/bin/sh\necho 'No Java runtime present' >&2\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    with pytest.raises(UnavailableError, match="exited 1"):
        probe_version(script)


def test_probe_version_refuses_a_missing_java(tmp_path):
    with pytest.raises(UnavailableError, match="-version failed"):
        probe_version(tmp_path / "absent")


def test_find_java_prefers_java_home(tmp_path, monkeypatch):
    java = tmp_path / "bin" / "java"
    java.parent.mkdir()
    java.write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/java")
    assert find_java() == java


def test_find_java_without_any_java_is_unavailable(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(UnavailableError, match="no java"):
        find_java()


def test_run_java_logs_output_and_returns_the_status(tmp_path):
    status = run_java([sys.executable, "-c", "print('hello'); raise SystemExit(4)"], tmp_path)
    assert status == 4
    assert "hello" in (tmp_path / "synthea.log").read_text(encoding="utf-8")


def test_run_java_stops_a_run_past_its_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_java([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=0.5)
