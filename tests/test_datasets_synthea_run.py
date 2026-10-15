"""Generating a pinned Synthea population: argv, Java, atomic directories, reuse."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from bko.datasets.errors import ContractError, DatasetError, UnavailableError
from bko.datasets.synthea_run import (
    CONFIG,
    JVM_OPTIONS,
    RUN_RECORD_NAME,
    SyntheaRun,
    check_population,
    default_java,
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
    head = [str(Path("/j").resolve()), *JVM_OPTIONS, "-jar", str(Path("/s.jar").resolve())]
    assert argv[: len(head)] == head
    assert "-Duser.timezone=UTC" in JVM_OPTIONS
    flags = argv[len(head) : len(head) + 10]
    pairs = dict(zip(flags[::2], flags[1::2], strict=True))
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
    record = json.loads((directory / RUN_RECORD_NAME).read_text(encoding="utf-8"))
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


@pytest.mark.parametrize(
    ("runner", "missing"),
    [
        (FakeSynthea(write=False), "patients.csv"),
        (FakeSynthea(omit="conditions.csv"), "conditions.csv"),
    ],
    ids=["no-tables", "no-conditions"],
)
def test_a_run_that_writes_a_table_short_is_refused_and_keeps_its_log(tmp_path, runner, missing):
    directory = population_dir(RUN, tmp_path, pin(JAR, "synthea-jar"))
    log = directory.with_name(directory.name + ".log")
    with pytest.raises(ContractError, match=f"wrote no {re.escape(missing)}.*its log is at"):
        generate(tmp_path, runner)
    assert log.read_text(encoding="utf-8") == "fake synthea\n"


def test_a_timeout_is_unavailable_and_keeps_the_log(tmp_path):
    def slow(argv, cwd):
        (cwd / "synthea.log").write_text("still going\n", encoding="utf-8")
        raise subprocess.TimeoutExpired(argv, 1.0)

    with pytest.raises(UnavailableError, match=r"did not finish.*its log is at"):
        generate(tmp_path, slow)
    directory = population_dir(RUN, tmp_path, pin(JAR, "synthea-jar"))
    log = directory.with_name(directory.name + ".log")
    assert log.read_text(encoding="utf-8") == "still going\n"


def test_a_failure_whose_log_is_missing_still_names_the_failure(tmp_path):
    def silent(argv, cwd):
        return 9

    with pytest.raises(UnavailableError, match=r"Synthea exited 9.*log could not be kept"):
        generate(tmp_path, silent)


def test_a_failed_run_leaves_no_staging_directory(tmp_path):
    with pytest.raises(UnavailableError):
        generate(tmp_path, FakeSynthea(status=3))
    assert [p.name for p in (tmp_path / "synthea").iterdir() if p.name.startswith(".")] == []


def test_a_success_removes_an_earlier_failure_log(tmp_path):
    with pytest.raises(UnavailableError):
        generate(tmp_path, FakeSynthea(status=3))
    directory = generate(tmp_path)
    assert not directory.with_name(directory.name + ".log").exists()


def test_the_population_key_names_every_parameter_and_the_jar():
    jar = pin(JAR, "synthea-jar")
    new_york = SyntheaRun(
        seed=7, clinician_seed=8, population=10, snapshot_date=date(2026, 1, 1), state="New York"
    )
    assert population_dir(new_york, Path("/d"), jar) == Path(
        f"/d/synthea/{jar.sha256[:16]}-s7-cs8-p10-d20260101-new-york"
    )


@pytest.mark.parametrize(
    "changes",
    [{"population": 0}, {"state": "../escape"}, {"state": "-p"}, {"state": "massachusetts"}],
    ids=["no-population", "path", "option", "lowercase"],
)
def test_a_run_with_a_bad_population_or_state_is_refused(changes):
    fields = {"seed": 1, "clinician_seed": 1, "population": 10, "snapshot_date": date(2026, 1, 1)}
    with pytest.raises(ValueError, match=r"population|state"):
        SyntheaRun(**(fields | changes))


@pytest.mark.parametrize(
    "record",
    [
        "{",
        "[]",
        '{"tables": []}',
        '{"tables": {}}',
        '{"tables": {"patients.csv": {"sha256": "x"}}}',
    ],
    ids=["not-json", "not-an-object", "tables-a-list", "no-tables", "one-table"],
)
def test_a_population_with_a_bad_record_is_regenerated(tmp_path, record):
    runner = FakeSynthea()
    directory = generate(tmp_path, runner)
    (directory / RUN_RECORD_NAME).write_text(record, encoding="utf-8")
    with pytest.raises(DatasetError):
        check_population(directory, RUN, pin(JAR, "synthea-jar"))
    generate(tmp_path, runner)
    assert len(runner.calls) == 2


@pytest.mark.parametrize("field", ["config", "jvm_options", "environment", "jar_sha256", "run"])
def test_a_population_recorded_with_other_settings_is_regenerated(tmp_path, field):
    runner = FakeSynthea()
    directory = generate(tmp_path, runner)
    path = directory / RUN_RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8")) | {field: "other"}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DatasetError, match=f"differs from this run in {field}"):
        check_population(directory, RUN, pin(JAR, "synthea-jar"))
    generate(tmp_path, runner)
    assert len(runner.calls) == 2


def test_a_population_a_concurrent_run_published_first_is_checked_and_used(tmp_path):
    directory = generate(tmp_path)
    saved = tmp_path / "saved"
    shutil.move(directory, saved)
    runner = FakeSynthea(then=lambda: shutil.copytree(saved, directory))
    assert generate(tmp_path, runner) == directory
    check_population(directory, RUN, pin(JAR, "synthea-jar"))


def test_java_is_found_before_the_jar_is_fetched(tmp_path):
    opener = FakeOpener(JAR)

    def no_java():
        raise UnavailableError("no runnable java")

    with pytest.raises(UnavailableError, match="no runnable java"):
        ensure_population(
            RUN, tmp_path, jar_source=pin(JAR, "synthea-jar"), open_url=opener, resolve_java=no_java
        )
    assert opener.calls == []


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
    java.chmod(0o755)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/java")
    assert find_java() == java.resolve()


def test_find_java_without_any_java_is_unavailable(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(UnavailableError, match="no runnable java on PATH"):
        find_java()


def test_run_java_logs_output_and_returns_the_status(tmp_path):
    status = run_java([sys.executable, "-c", "print('hello'); raise SystemExit(4)"], tmp_path)
    assert status == 4
    assert "hello" in (tmp_path / "synthea.log").read_text(encoding="utf-8")


def test_run_java_stops_a_run_past_its_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_java([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=0.5)


def fake_java(directory, body):
    """Write an executable `bin/java` shell script under `directory` and return it."""
    java = directory / "bin" / "java"
    java.parent.mkdir(parents=True, exist_ok=True)
    java.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    java.chmod(0o755)
    return java


def test_a_java_home_without_java_does_not_fall_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/java")
    expected = re.escape(f"JAVA_HOME ({tmp_path.resolve()}/bin/java)")
    with pytest.raises(UnavailableError, match=expected):
        find_java()


def test_a_java_that_is_not_executable_is_refused(tmp_path, monkeypatch):
    java = tmp_path / "bin" / "java"
    java.parent.mkdir()
    java.write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    with pytest.raises(UnavailableError, match="no runnable java"):
        find_java()


def test_probe_version_skips_a_banner_and_ignores_java_options(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx1g")
    body = (
        'echo "Picked up JAVA_TOOL_OPTIONS: $JAVA_TOOL_OPTIONS" >&2\n'
        "echo 'openjdk version \"21.0.4\"' >&2\n"
    )
    assert probe_version(fake_java(tmp_path, body)) == 'openjdk version "21.0.4"'


def test_default_java_returns_java_home_java_and_its_version(tmp_path, monkeypatch):
    java = fake_java(tmp_path, "echo 'openjdk version \"17.0.19\"' >&2\n")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    assert default_java() == (java.resolve(), 'openjdk version "17.0.19"')


def test_run_java_runs_in_cwd_with_only_the_pinned_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BKO_SECRET", "leak")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    code = (
        "import os, sys; "
        "print(os.getcwd(), os.environ.get('TZ'), os.environ.get('LANG'), "
        "os.environ.get('BKO_SECRET')); "
        "print('on stderr', file=sys.stderr)"
    )
    assert run_java([sys.executable, "-c", code], tmp_path) == 0
    log = (tmp_path / "synthea.log").read_text(encoding="utf-8")
    assert f"{os.path.realpath(tmp_path)} UTC C.UTF-8 None" in log
    assert "on stderr" in log
