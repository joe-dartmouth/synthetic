"""`python -m bko.evaluate launch`: refuses on battery, writes the roles, starts detached."""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from bko.app.runs import ROLES
from bko.evaluate import launch as launching
from bko.evaluate.__main__ import Context, main
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, RUNS, NamedRun
from bko.evaluate.launch import REHEARSAL, TAKEN, launch, on_battery, spawn


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


class Starts:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command, out, err) -> int:
        self.commands.append(tuple(command))
        return 4242


def test_it_refuses_on_battery_power_and_writes_nothing(tmp_path):
    start = Starts()
    with pytest.raises(EvaluationError, match="battery power"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: True, start=start)
    assert not (tmp_path / "e1-4").exists()
    assert start.commands == []


def test_a_dry_run_writes_its_log_starts_nothing_and_leaves_the_folder_to_launch(tmp_path):
    start = Starts()
    found = launch("e1-4", "e1", root=tmp_path, dry_run=True, power=lambda: False, start=start)
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == ["launch.log"]
    assert start.commands == []
    assert not found.started
    assert found.pid is None
    assert found.watch == f"tail -f {folder / 'scores.log'}"
    log = (folder / "launch.log").read_text(encoding="utf-8")
    assert "dry_run=True" in log
    assert "power=mains" in log
    assert log.endswith("pid=None\n")
    assert launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=start).started
    assert sorted(path.name for path in folder.iterdir()) == [
        "launch.log",
        "scores.csv",
        "scores.log",
    ]


def test_it_starts_score_with_its_call_log_kept_awake_on_macos(tmp_path):
    start = Starts()
    found = launch(
        "e1-4",
        "e1",
        root=tmp_path,
        data_dir=tmp_path / "data",
        power=lambda: False,
        start=start,
        platform="darwin",
    )
    (command,) = start.commands
    assert command[:3] == ("caffeinate", "-i", "-s")
    assert command[3:7] == (sys.executable, "-m", "bko.evaluate", "score")
    assert command[7:] == (
        "e1",
        "--log",
        str(tmp_path / "e1-4" / "calls.jsonl"),
        "--candidates",
        str(tmp_path / "e1-4" / "holdout-candidates.jsonl"),
        "--data-dir",
        str(tmp_path / "data"),
    )
    assert found.started
    assert found.pid == 4242
    log = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8").splitlines()
    assert log[1:] == [
        "power=mains",
        "awake=caffeinate",
        (
            f"command=caffeinate -i -s {sys.executable} -m bko.evaluate score e1 --log "
            f"{tmp_path / 'e1-4' / 'calls.jsonl'} --candidates "
            f"{tmp_path / 'e1-4' / 'holdout-candidates.jsonl'} --data-dir {tmp_path / 'data'}"
        ),
        "pid=4242",
    ]


def test_on_macos_a_power_reading_that_cannot_be_read_refuses(tmp_path):
    start = Starts()
    with pytest.raises(EvaluationError, match="mains power cannot be confirmed"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: None, start=start, platform="darwin")
    assert not (tmp_path / "e1-4").exists()
    assert start.commands == []


def test_a_process_that_cannot_start_leaves_the_folder_free_and_records_why(tmp_path):
    def missing(command, out, err):
        raise FileNotFoundError(2, "No such file or directory", command[0])

    with pytest.raises(EvaluationError, match="caffeinate could not start"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=missing, platform="darwin")
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == ["launch.log"]
    assert "error=FileNotFoundError" in (folder / "launch.log").read_text(encoding="utf-8")
    assert launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=Starts()).started
    log = (folder / "launch.log").read_text(encoding="utf-8")
    assert "error=FileNotFoundError\n" in log
    assert log.count("launched ") == 2
    assert log.endswith("pid=4242\n")


def test_a_start_that_fails_otherwise_also_leaves_the_folder_free(tmp_path):
    def broken(command, out, err):
        raise RuntimeError

    with pytest.raises(RuntimeError):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=broken)
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == ["launch.log"]
    assert "error=RuntimeError" in (folder / "launch.log").read_text(encoding="utf-8")


def test_a_launch_log_that_cannot_be_written_leaves_the_folder_free(tmp_path, monkeypatch):
    def unwritable(record, lines):
        raise PermissionError(13, "Permission denied", str(record))

    monkeypatch.setattr(launching, "_append", unwritable)
    start = Starts()
    with pytest.raises(PermissionError):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=start)
    assert list((tmp_path / "e1-4").iterdir()) == []
    assert start.commands == []


def test_a_claim_that_fails_otherwise_removes_what_it_made(tmp_path, monkeypatch):
    opening = pathlib.Path.open

    def refusing(self, *args, **kwargs):
        if self.name == "scores.log":
            raise PermissionError(13, "Permission denied", str(self))
        return opening(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", refusing)
    start = Starts()
    with pytest.raises(EvaluationError, match="could not claim"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=start)
    assert list((tmp_path / "e1-4").iterdir()) == []
    assert start.commands == []


def test_a_data_root_that_reads_as_an_option_is_refused_before_the_folder(tmp_path, caplog):
    with pytest.raises(EvaluationError, match="reads as an option"):
        launch(
            "e1-4", "e1", root=tmp_path, data_dir=Path("-x"), power=lambda: False, start=Starts()
        )
    context = Context(power=lambda: False, runs_dir=tmp_path, start=Starts())
    with caplog.at_level(logging.ERROR):
        assert main(["launch", "e1-4", "--data-dir=-x"], context=context) == 2
    assert "reads as an option" in caplog.text
    assert not (tmp_path / "e1-4").exists()


def _stopped(tmp_path: Path) -> Path:
    source = tmp_path / "g3-1" / "calls.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    return source


def test_a_resume_starts_score_on_the_stopped_log_into_a_new_folder(tmp_path):
    source = _stopped(tmp_path)
    context = named(tmp_path, {**RUNS, "g3-cell": NamedRun(E1)})
    argv = ["launch", "g3-6", "g3-cell", "--resume", str(source)]
    assert main(argv, context=context, out=io.StringIO()) == 0
    folder = tmp_path / "g3-6"
    (command,) = context.start.commands
    assert command[-8:] == (
        "score",
        "g3-cell",
        "--resume",
        str(source.absolute()),
        "--log",
        str(folder / "calls.jsonl"),
        "--candidates",
        str(folder / "holdout-candidates.jsonl"),
    )
    log = (folder / "launch.log").read_text(encoding="utf-8")
    assert f"resume={source.absolute()}" in log


def test_a_resume_source_inside_the_target_folder_is_refused(tmp_path):
    source = tmp_path / "g3-6" / "old.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    start = Starts()
    with pytest.raises(EvaluationError, match="inside"):
        launch("g3-6", "g3", root=tmp_path, resume=source, power=lambda: False, start=start)
    assert sorted(path.name for path in source.parent.iterdir()) == ["old.jsonl"]
    assert start.commands == []


def test_a_resume_source_that_is_not_a_file_is_refused(tmp_path):
    start = Starts()
    with pytest.raises(EvaluationError, match="not a file"):
        launch(
            "g3-6",
            "g3",
            root=tmp_path,
            resume=tmp_path / "missing.jsonl",
            power=lambda: False,
            start=start,
        )
    assert not (tmp_path / "g3-6").exists()


def test_a_resume_keeps_the_power_and_folder_checks(tmp_path):
    source = _stopped(tmp_path)
    with pytest.raises(EvaluationError, match="battery power"):
        launch("g3-6", "g3", root=tmp_path, resume=source, power=lambda: True, start=Starts())
    (tmp_path / "g3-6").mkdir()
    (tmp_path / "g3-6" / "calls.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("g3-6", "g3", root=tmp_path, resume=source, power=lambda: False, start=Starts())


def test_rehearse_and_resume_are_exclusive(tmp_path):
    source = _stopped(tmp_path)
    context = Context(power=lambda: False, runs_dir=tmp_path, start=Starts())
    with pytest.raises(SystemExit) as exited:
        main(["launch", "e1-4", "--rehearse", "--resume", str(source)], context=context)
    assert exited.value.code == 2
    assert not (tmp_path / "e1-4").exists()
    with pytest.raises(EvaluationError, match="resumes nothing"):
        launch("e1-4", "e1", root=tmp_path, rehearse=True, resume=source, power=lambda: False)
    assert not (tmp_path / "e1-4").exists()


def test_a_launch_that_loses_the_folder_to_another_starts_nothing(tmp_path, monkeypatch):
    # Another launch created its output between this one's check and its claim.
    monkeypatch.setattr(launching, "TAKEN", ())
    folder = tmp_path / "e1-4"
    folder.mkdir()
    (folder / "scores.log").write_text("theirs", encoding="utf-8")
    start = Starts()
    with pytest.raises(EvaluationError, match="another launch started first"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=start)
    assert start.commands == []
    assert sorted(path.name for path in folder.iterdir()) == ["scores.log"]
    assert (folder / "scores.log").read_text(encoding="utf-8") == "theirs"


def test_the_roles_a_launch_guards_are_run_folder_roles():
    assert set(TAKEN) <= set(ROLES)
    assert "launch.log" in ROLES
    assert REHEARSAL in ROLES


def test_a_rehearsal_launches_into_its_folder_inside_the_runs_and_leaves_the_run_free(tmp_path):
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start)
    assert main(["launch", "e1-4", "--rehearse"], context=context, out=io.StringIO()) == 0
    folder = tmp_path / "e1-4" / REHEARSAL
    (command,) = start.commands
    assert command[-1] == "--rehearse"
    assert str(folder / "calls.jsonl") in command
    assert str(folder / "holdout-candidates.jsonl") in command
    assert "rehearse=true" in (folder / "launch.log").read_text(encoding="utf-8")
    assert sorted(path.name for path in (tmp_path / "e1-4").iterdir()) == [REHEARSAL]
    assert main(["launch", "e1-4"], context=context, out=io.StringIO()) == 0
    assert "--rehearse" not in start.commands[-1]
    assert str(tmp_path / "e1-4" / "calls.jsonl") in start.commands[-1]
    assert "rehearse=false" in (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")


def test_a_second_rehearsal_into_one_folder_is_refused(tmp_path):
    launch("e1-4", "e1", root=tmp_path, rehearse=True, power=lambda: False, start=Starts())
    (tmp_path / "e1-4" / REHEARSAL / "calls.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("e1-4", "e1", root=tmp_path, rehearse=True, power=lambda: False, start=Starts())


def test_spawn_starts_the_command_detached_with_its_output_in_the_files(tmp_path):
    out, err = tmp_path / "out", tmp_path / "err"
    code = "import os, sys; print(os.getsid(0) == os.getpid()); print('e', file=sys.stderr)"
    pid = spawn([sys.executable, "-c", code], out, err)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert out.read_text(encoding="utf-8") == "True\n"
    assert err.read_text(encoding="utf-8") == "e\n"


def test_elsewhere_nothing_holds_the_machine_awake_and_power_is_recorded_unchecked(tmp_path):
    start = Starts()
    launch("e1-4", "e1", root=tmp_path, power=lambda: None, start=start, platform="linux")
    assert start.commands[0][0] == sys.executable
    log = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")
    assert "awake=none" in log
    assert "power=unchecked" in log


@pytest.mark.parametrize("name", TAKEN)
def test_it_refuses_a_folder_that_already_holds_a_run(tmp_path, name):
    (tmp_path / "e1-4").mkdir()
    (tmp_path / "e1-4" / name).write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=Starts())


@pytest.mark.parametrize("run_id", ["e1", "../e1-4", "app-20260101T000000000000Z-bko", "E1-4"])
def test_it_refuses_an_id_that_is_not_a_run_of_record(tmp_path, run_id):
    with pytest.raises(EvaluationError, match="not a run id of record"):
        launch(run_id, "e1", root=tmp_path, power=lambda: False, start=Starts())


def test_it_refuses_an_id_of_another_run(tmp_path):
    with pytest.raises(EvaluationError, match="not a run id of the run 'e1'"):
        launch("e2-4", "e1", root=tmp_path, power=lambda: False, start=Starts())
    assert not (tmp_path / "e2-4").exists()


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Now drawing from 'Battery Power'\n -InternalBattery-0\t24%; discharging", True),
        ("Now drawing from 'AC Power'\n -InternalBattery-0\t24%; charging", False),
    ],
)
def test_the_power_reading_comes_from_pmset(monkeypatch, stdout, expected):
    monkeypatch.setattr(launching.sys, "platform", "darwin")
    monkeypatch.setattr(
        launching.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout),
    )
    assert on_battery() is expected


def test_the_power_reading_is_none_when_it_cannot_be_checked(monkeypatch):
    monkeypatch.setattr(launching.sys, "platform", "linux")
    assert on_battery() is None
    monkeypatch.setattr(launching.sys, "platform", "darwin")

    def missing(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(launching.subprocess, "run", missing)
    assert on_battery() is None
    monkeypatch.setattr(
        launching.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="")
    )
    assert on_battery() is None

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("pmset", 30)

    monkeypatch.setattr(launching.subprocess, "run", slow)
    assert on_battery() is None


def test_the_command_prints_the_watch_command_and_exits_zero_on_a_dry_run(tmp_path):
    out = io.StringIO()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=Starts())
    assert main(["launch", "e1-4", "--dry-run"], context=context, out=out) == 0
    assert out.getvalue() == f"tail -f {tmp_path / 'e1-4' / 'scores.log'}\n"


def test_the_command_exits_two_on_battery_power(tmp_path, caplog):
    out = io.StringIO()
    context = Context(power=lambda: True, runs_dir=tmp_path, start=Starts())
    with caplog.at_level(logging.ERROR):
        assert main(["launch", "e1-4"], context=context, out=out) == 2
    assert "battery power" in caplog.text
    assert out.getvalue() == ""


def test_the_command_refuses_an_unknown_run_name(tmp_path):
    out = io.StringIO()
    context = Context(power=lambda: False, runs_dir=tmp_path)
    assert main(["launch", "e1-4", "nope"], context=context, out=out) == 2
    assert not (tmp_path / "e1-4").exists()


def test_the_command_refuses_a_run_with_no_bko_arm(tmp_path, caplog):
    unpaid = dataclasses.replace(
        E1, arms=("none", "smote"), candidate="smote", comparators=("none",), replica_arms=()
    )
    start = Starts()
    context = Context(experiment=unpaid, power=lambda: False, runs_dir=tmp_path, start=start)
    with caplog.at_level(logging.ERROR):
        assert main(["launch", "e1-4"], context=context, out=io.StringIO()) == 2
    assert "has no BKO arm" in caplog.text
    assert start.commands == []
    assert not (tmp_path / "e1-4").exists()


def named(tmp_path, runs):
    return Context(power=lambda: False, runs_dir=tmp_path, start=Starts(), runs=runs)


def test_the_run_defaults_to_the_run_ids_entry(tmp_path):
    context = named(tmp_path, {**RUNS, "e2": NamedRun(E1)})
    assert main(["launch", "e2-4", "--dry-run"], context=context, out=io.StringIO()) == 0
    assert "run=e2 " in (tmp_path / "e2-4" / "launch.log").read_text(encoding="utf-8")
    assert main(["launch", "e1-4", "--dry-run"], context=context, out=io.StringIO()) == 0
    assert "run=e1 " in (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")


def test_a_named_run_of_the_run_ids_entry_launches(tmp_path):
    context = named(tmp_path, {**RUNS, "g3-cell": NamedRun(E1)})
    assert main(["launch", "g3-1", "g3-cell", "--dry-run"], context=context, out=io.StringIO()) == 0
    log = (tmp_path / "g3-1" / "launch.log").read_text(encoding="utf-8")
    assert "run=g3-cell " in log
    assert " score g3-cell --log " in log


def test_a_named_run_of_another_entry_is_refused(tmp_path, caplog):
    context = named(tmp_path, {**RUNS, "g3-cell": NamedRun(E1)})
    with caplog.at_level(logging.ERROR):
        assert main(["launch", "g3-1", "e1"], context=context, out=io.StringIO()) == 2
    assert "not a run id of the run 'e1'" in caplog.text
    assert not (tmp_path / "g3-1").exists()
    with pytest.raises(EvaluationError, match="not a run id of the run 'e1'"):
        launch("g3-1", "e1", root=tmp_path, power=lambda: False, start=Starts())


def test_a_named_run_whose_entry_only_starts_with_the_run_ids_is_refused(tmp_path):
    context = named(tmp_path, {**RUNS, "e10": NamedRun(E1)})
    assert main(["launch", "e1-4", "e10", "--dry-run"], context=context, out=io.StringIO()) == 2
    assert not (tmp_path / "e1-4").exists()


@pytest.mark.parametrize("run_id", ["nope", "app-20260101T000000000000Z-bko", "../e1-4"])
def test_an_id_that_is_not_a_run_of_record_exits_two(tmp_path, caplog, run_id):
    with caplog.at_level(logging.ERROR):
        assert main(["launch", run_id], context=named(tmp_path, RUNS), out=io.StringIO()) == 2
    assert "not a run id of record" in caplog.text


def test_every_named_run_is_an_entry_or_a_word_after_it():
    assert all(re.fullmatch(r"[a-z]\d+(?:-[a-z]+)?", name) for name in RUNS)
