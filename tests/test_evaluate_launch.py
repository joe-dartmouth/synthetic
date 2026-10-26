"""`python -m bko.evaluate launch`: refuses on battery, writes the roles, starts detached."""

from __future__ import annotations

import io
import logging
import subprocess
import sys

import pytest

from bko.evaluate import launch as launching
from bko.evaluate.__main__ import Context, main
from bko.evaluate.errors import EvaluationError
from bko.evaluate.launch import launch, on_battery


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


def test_a_dry_run_writes_the_folders_roles_and_starts_nothing(tmp_path):
    start = Starts()
    found = launch("e1-4", "e1", root=tmp_path, dry_run=True, power=lambda: False, start=start)
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == [
        "launch.log",
        "scores.csv",
        "scores.log",
    ]
    assert not (folder / "calls.jsonl").exists()
    assert start.commands == []
    assert not found.started
    assert found.pid is None
    assert found.watch == f"tail -f {folder / 'scores.log'}"
    log = (folder / "launch.log").read_text(encoding="utf-8")
    assert "dry_run=True" in log
    assert "power=mains" in log


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
    assert command[:2] == ("caffeinate", "-i")
    assert command[2:6] == (sys.executable, "-m", "bko.evaluate", "score")
    assert command[6:] == (
        "e1",
        "--log",
        str(tmp_path / "e1-4" / "calls.jsonl"),
        "--data-dir",
        str(tmp_path / "data"),
    )
    assert found.started
    assert found.pid == 4242
    log = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")
    assert "awake=caffeinate" in log
    assert "pid=4242" in log


def test_elsewhere_nothing_holds_the_machine_awake_and_power_is_recorded_unchecked(tmp_path):
    start = Starts()
    launch("e1-4", "e1", root=tmp_path, power=lambda: None, start=start, platform="linux")
    assert start.commands[0][0] == sys.executable
    log = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")
    assert "awake=none" in log
    assert "power=unchecked" in log


@pytest.mark.parametrize("name", ["calls.jsonl", "scores.csv", "scores.log"])
def test_it_refuses_a_folder_that_already_holds_a_run(tmp_path, name):
    (tmp_path / "e1-4").mkdir()
    (tmp_path / "e1-4" / name).write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=Starts())


@pytest.mark.parametrize("run_id", ["e1", "../e1-4", "app-20260101T000000000000Z-bko", "E1-4"])
def test_it_refuses_an_id_that_is_not_a_run_of_record(tmp_path, run_id):
    with pytest.raises(EvaluationError, match="not a run id of record"):
        launch(run_id, "e1", root=tmp_path, power=lambda: False, start=Starts())


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
