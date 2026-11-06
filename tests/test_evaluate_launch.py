"""`python -m bko.evaluate launch`: refuses on battery, writes the roles, starts detached."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import pytest

from bko.app.runs import ROLES
from bko.datasets.run_folders import RUN_ID
from bko.evaluate import launch as launching
from bko.evaluate.__main__ import Context, main
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1
from bko.evaluate.launch import REHEARSAL, TAKEN, launch, on_battery, spawn
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.generate.log import Answered, CallLog
from bko.generate.messages import SETTINGS, Level


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
    assert command[3:7] == (sys.executable, "-m", "bko.evaluate", "supervise")
    assert command[7:] == (
        "e1-4",
        "e1",
        "--retries",
        "5",
        "--runs-dir",
        str(tmp_path.absolute()),
        "--data-dir",
        str((tmp_path / "data").absolute()),
    )
    assert found.started
    assert found.pid == 4242
    log = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8").splitlines()
    assert log[1:] == [
        "power=mains",
        "awake=caffeinate",
        (
            f"command=caffeinate -i -s {sys.executable} -m bko.evaluate supervise e1-4 e1 "
            f"--retries 5 --runs-dir {tmp_path} --data-dir {tmp_path / 'data'}"
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
    assert "error=FileNotFoundError: [Errno 2] No such file or directory: 'caffeinate'\n" in log
    assert log.count("launched ") == 2
    assert log.endswith("pid=4242\n")


def test_a_start_that_fails_otherwise_also_leaves_the_folder_free(tmp_path):
    def broken(command, out, err):
        raise RuntimeError

    with pytest.raises(RuntimeError):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=broken)
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == ["launch.log"]
    assert "error=RuntimeError: \n" in (folder / "launch.log").read_text(encoding="utf-8")


def _failing_on(prefix: str, monkeypatch) -> None:
    """Make `_append` raise for the lines that start with `prefix`, and write the others."""
    appending = launching.append

    def failing(record, lines):
        if lines and lines[0].startswith(prefix):
            raise PermissionError(13, "Permission denied", str(record))
        appending(record, lines)

    monkeypatch.setattr(launching, "append", failing)


def test_a_started_process_whose_pid_line_fails_is_reported_started(tmp_path, monkeypatch, caplog):
    _failing_on("pid=", monkeypatch)
    with caplog.at_level(logging.WARNING):
        found = launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=Starts())
    assert found.started
    assert found.pid == 4242
    assert "launch pid=4242 unrecorded error=PermissionError" in caplog.text
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == [
        "launch.log",
        "scores.csv",
        "scores.log",
    ]


def test_an_error_line_that_cannot_be_written_is_warned_and_the_folder_freed(
    tmp_path, monkeypatch, caplog
):
    def missing(command, out, err):
        raise FileNotFoundError(2, "No such file or directory", command[0])

    _failing_on("error=", monkeypatch)
    with caplog.at_level(logging.WARNING), pytest.raises(EvaluationError, match="could not start"):
        launch("e1-4", "e1", root=tmp_path, power=lambda: False, start=missing)
    assert "launch error unrecorded error=PermissionError" in caplog.text
    folder = tmp_path / "e1-4"
    assert sorted(path.name for path in folder.iterdir()) == ["launch.log"]


def test_a_launch_log_that_cannot_be_written_leaves_the_folder_free(tmp_path, monkeypatch):
    def unwritable(record, lines):
        raise PermissionError(13, "Permission denied", str(record))

    monkeypatch.setattr(launching, "append", unwritable)
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


def test_a_data_root_is_forwarded_absolute_so_a_leading_dash_never_reads_as_an_option(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start)
    assert main(["launch", "e1-4", "--data-dir=-x"], context=context, out=io.StringIO()) == 0
    (command,) = start.commands
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "-x")


def _stopped(
    tmp_path: Path,
    *,
    stop: str | None = "idle",
    rehearsal: bool = False,
    settings=SETTINGS,
    run: str | None = "g9-probe",
    **end: object,
) -> Path:
    """A stopped call log of the run `run` under `g9-1`, its end line naming `stop` and `end`."""
    source = tmp_path / "g9-1" / "calls.jsonl"
    source.parent.mkdir(exist_ok=True)
    with CallLog(source, settings, rehearsal=rehearsal, run=run, frame="uci"):
        pass
    lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    lines[-1].update(status="failed", stop=stop, **end)
    source.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return source


SPENT: Final[dict[str, Any]] = {"usd": 1.2, "live_usd": 1.2, "holdouts": 5}
"""A stopped log's spend: 1.2 USD over 5 of E1's 100 holdouts, which projects to 24.0 USD."""


@pytest.mark.parametrize(
    ("flags", "raised", "message"),
    [
        ({"rehearsal": True}, None, "rehearsal"),
        ({"stop": "projection"}, None, "resumes only under a ceiling raised past its projection"),
        ({"stop": "ceiling", **SPENT}, None, "resume it with launch --resume and --ceiling"),
        ({"stop": "projection"}, "30", "records no finished holdouts"),
        ({"stop": "ceiling"}, "30", "records no finished holdouts"),
        ({"stop": "ceiling", "usd": 1.5, "holdouts": 5}, "30", "projects to 30.0000 USD over 100"),
        (
            {"stop": "projection", "usd": 1.25, "holdouts": 5},
            "25",
            "raised ceiling of 25.0000 USD does not",
        ),
        ({"stop": "ceiling", **SPENT, "usd": -1.0}, "30", "no usd and live_usd spend of at least"),
        ({"stop": "ceiling", **SPENT, "run": None}, "30", "names no run, so it cannot be checked"),
        ({"run": "g9-other"}, None, "holds the calls of the run 'g9-other', not 'g9-probe'"),
        ({"stop": "idle"}, "30", "takes no raised ceiling; resume it without --ceiling"),
        ({"settings": dataclasses.replace(SETTINGS, guidance=Level.CELL)}, None, "settings"),
    ],
)
def test_a_resume_refuses_a_log_it_cannot_resume_before_the_folder(
    tmp_path, caplog, flags, raised, message
):
    source = _stopped(tmp_path, **flags)
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-6", "g9-probe", "--resume", str(source)]
    with caplog.at_level(logging.ERROR):
        code = main([*argv, *(["--ceiling", raised] if raised else [])], context=context)
    assert code == 2
    assert message in caplog.text
    assert not (tmp_path / "g9-6").exists()
    assert context.start.commands == []


def test_a_resume_supervises_the_stopped_log_into_a_new_folder(tmp_path):
    source = _stopped(tmp_path)
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-6", "g9-probe", "--resume", str(source)]
    assert main(argv, context=context, out=io.StringIO()) == 0
    folder = tmp_path / "g9-6"
    (command,) = context.start.commands
    assert command[-9:] == (
        "supervise",
        "g9-6",
        "g9-probe",
        "--retries",
        "5",
        "--runs-dir",
        str(tmp_path.absolute()),
        "--resume",
        str(source.absolute()),
    )
    log = (folder / "launch.log").read_text(encoding="utf-8")
    assert f"resume={source.absolute()}" in log


@pytest.mark.parametrize("stop", ["projection", "ceiling"])
def test_a_spend_stop_resumes_under_a_raised_ceiling_that_covers_its_projection(tmp_path, stop):
    # 1.2 USD over 5 of E1's 100 holdouts projects to 24.0 USD; the raised 24.5 covers it.
    source = _stopped(tmp_path, stop=stop, **SPENT)
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-6", "g9-probe", "--resume", str(source), "--ceiling", "24.5"]
    assert main(argv, context=context, out=io.StringIO()) == 0
    (command,) = context.start.commands
    assert command[-4:] == ("--resume", str(source.absolute()), "--ceiling", "24.5")
    log = (tmp_path / "g9-6" / "launch.log").read_text(encoding="utf-8").splitlines()
    assert log[1] == (
        f"ceiling=24.5000 stop={stop} usd=1.2000 live_usd=1.2000 holdouts=5 projected_usd=24.0000"
    )


def test_a_ceiling_without_a_resume_is_refused(tmp_path, caplog):
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    with caplog.at_level(logging.ERROR):
        code = main(["launch", "g9-6", "g9-probe", "--ceiling", "30"], context=context)
    assert code == 2
    assert "--ceiling raises a resumed run's spend ceiling, so it needs --resume" in caplog.text
    assert not (tmp_path / "g9-6").exists()


@pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "x"])
def test_a_ceiling_that_is_no_spend_above_zero_is_refused(tmp_path, capsys, text):
    source = _stopped(tmp_path, stop="ceiling", **SPENT)
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-6", "g9-probe", "--resume", str(source), f"--ceiling={text}"]
    with pytest.raises(SystemExit):
        main(argv, context=context)
    assert "the ceiling must be a finite number of USD above 0" in capsys.readouterr().err
    assert context.start.commands == []


def test_a_resume_refuses_a_log_a_later_segment_already_resumed(tmp_path, caplog):
    source = _stopped(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    later = tmp_path / "g9-1-resume-1" / "calls.jsonl"
    later.parent.mkdir()
    with CallLog(
        later, SETTINGS, answered=Answered("resume", digest, 0), run="g9-probe", frame="uci"
    ):
        pass
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    with caplog.at_level(logging.ERROR):
        code = main(["launch", "g9-6", "g9-probe", "--resume", str(source)], context=context)
    assert code == 2
    assert f"{later} already resumed {source} (sha256 {digest})" in caplog.text
    assert "resume the last segment's log" in caplog.text
    assert not (tmp_path / "g9-6").exists()
    assert context.start.commands == []
    argv = ["launch", "g9-6", "g9-probe", "--resume", str(later)]
    assert main(argv, context=context, out=io.StringIO()) == 0


def test_a_resume_source_inside_the_target_folder_is_refused(tmp_path):
    source = tmp_path / "g9-6" / "old.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    start = Starts()
    with pytest.raises(EvaluationError, match="inside"):
        launch("g9-6", "g9", root=tmp_path, resume=source, power=lambda: False, start=start)
    assert sorted(path.name for path in source.parent.iterdir()) == ["old.jsonl"]
    assert start.commands == []


def test_a_relative_resume_source_is_forwarded_absolute(tmp_path, monkeypatch):
    _stopped(tmp_path)
    monkeypatch.chdir(tmp_path)
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-6", "g9-probe", "--resume", "g9-1/calls.jsonl"]
    assert main(argv, context=context, out=io.StringIO()) == 0
    (command,) = context.start.commands
    assert command[command.index("--resume") + 1] == str(tmp_path / "g9-1" / "calls.jsonl")


def test_a_resume_source_in_a_subfolder_of_the_target_folder_is_refused(tmp_path):
    source = tmp_path / "g9-6" / "deep" / "old.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    start = Starts()
    with pytest.raises(EvaluationError, match="inside"):
        launch("g9-6", "g9", root=tmp_path, resume=source, power=lambda: False, start=start)
    assert start.commands == []
    assert not (tmp_path / "g9-6" / "scores.log").exists()


def test_a_resume_source_that_is_not_a_file_is_refused(tmp_path):
    start = Starts()
    with pytest.raises(EvaluationError, match="not a file"):
        launch(
            "g9-6",
            "g9",
            root=tmp_path,
            resume=tmp_path / "missing.jsonl",
            power=lambda: False,
            start=start,
        )
    assert not (tmp_path / "g9-6").exists()


def test_a_resume_keeps_the_power_and_folder_checks(tmp_path):
    source = _stopped(tmp_path)
    with pytest.raises(EvaluationError, match="battery power"):
        launch("g9-6", "g9", root=tmp_path, resume=source, power=lambda: True, start=Starts())
    (tmp_path / "g9-6").mkdir()
    (tmp_path / "g9-6" / "calls.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("g9-6", "g9", root=tmp_path, resume=source, power=lambda: False, start=Starts())


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
    assert set(TAKEN) == {"calls.jsonl", "scores.csv", "scores.log", "holdout-candidates.jsonl"}
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
    assert command[command.index("supervise") + 1 :][:2] == ("e1-4", "e1")
    assert "rehearse=true" in (folder / "launch.log").read_text(encoding="utf-8")
    assert sorted(path.name for path in (tmp_path / "e1-4").iterdir()) == [REHEARSAL]
    assert main(["launch", "e1-4"], context=context, out=io.StringIO()) == 0
    assert "--rehearse" not in start.commands[-1]
    assert "rehearse=false" in (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8")


def test_launch_forwards_in_flight_when_given(tmp_path):
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start)
    assert main(["launch", "e1-4", "--in-flight", "16"], context=context, out=io.StringIO()) == 0
    (command,) = start.commands
    assert command[-2:] == ("--in-flight", "16")
    assert main(["launch", "e1-5"], context=context, out=io.StringIO()) == 0
    assert "--in-flight" not in start.commands[-1]


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_launch_refuses_an_in_flight_below_one(tmp_path, value):
    context = Context(power=lambda: False, runs_dir=tmp_path, start=Starts())
    with pytest.raises(SystemExit):
        main(["launch", "e1-4", "--in-flight", value], context=context, out=io.StringIO())


def test_a_second_rehearsal_into_one_folder_is_refused(tmp_path):
    launch("e1-4", "e1", root=tmp_path, rehearse=True, power=lambda: False, start=Starts())
    (tmp_path / "e1-4" / REHEARSAL / "calls.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already holds"):
        launch("e1-4", "e1", root=tmp_path, rehearse=True, power=lambda: False, start=Starts())


SESSION_PROBE = "import os, sys; print(os.getsid(0) == os.getpid()); print('e', file=sys.stderr)"
"""A child that prints whether it leads its own session, then a line to stderr."""


def _written(path: Path, text: str, *, timeout: float = 10.0) -> bool:
    """Whether `path` holds `text` within `timeout` seconds; nothing waits on the spawned child."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return True
        time.sleep(0.01)
    return False


def test_spawn_starts_the_command_detached_with_its_output_in_the_files(tmp_path):
    out, err = tmp_path / "out", tmp_path / "err"
    spawn([sys.executable, "-c", SESSION_PROBE], out, err)
    assert _written(out, "True\n")
    assert _written(err, "e\n")


class _ExitedPopen(subprocess.Popen):
    """A `Popen` that returns only once its child has exited, leaving the child unreaped."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)


def test_spawn_leaves_no_child_for_the_caller_to_reap(tmp_path, monkeypatch):
    """The race `os.waitpid` lost: a child that exits before `spawn` returns is reaped when the
    dropped `Popen` is collected, so a caller waiting on the pid finds no child."""
    monkeypatch.setattr(launching.subprocess, "Popen", _ExitedPopen)
    out, err = tmp_path / "out", tmp_path / "err"
    pid = spawn([sys.executable, "-c", SESSION_PROBE], out, err)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert out.read_text(encoding="utf-8") == "True\n"


def test_launch_passes_retries(tmp_path):
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start)
    assert main(["launch", "e1-4", "--retries", "0"], context=context, out=io.StringIO()) == 0
    assert main(["launch", "e1-5"], context=context, out=io.StringIO()) == 0
    once, default = start.commands
    assert once[once.index("--retries") + 1] == "0"
    assert default[default.index("--retries") + 1] == "5"
    assert default[default.index("--runs-dir") + 1] == str(tmp_path.absolute())
    with pytest.raises(SystemExit):
        main(["launch", "e1-6", "--retries", "-1"], context=context, out=io.StringIO())
    with pytest.raises(SystemExit):
        main(["launch", "e1-6", "--retries", "many"], context=context, out=io.StringIO())


@pytest.mark.parametrize("command", ["launch", "supervise"])
def test_retries_are_bounded_at_twenty(tmp_path, capsys, command):
    context = Context(power=lambda: False, runs_dir=tmp_path, start=Starts())
    tail = ["e1", "--runs-dir", str(tmp_path)] if command == "supervise" else []
    with pytest.raises(SystemExit):
        main([command, "e1-6", *tail, "--retries", "21"], context=context, out=io.StringIO())
    assert "at most 20, not '21'" in capsys.readouterr().err
    assert main(["launch", "e1-6", "--retries", "20"], context=context, out=io.StringIO()) == 0


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
    assert "has no paid arm" in caplog.text
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
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    argv = ["launch", "g9-1", "g9-probe", "--dry-run"]
    assert main(argv, context=context, out=io.StringIO()) == 0
    log = (tmp_path / "g9-1" / "launch.log").read_text(encoding="utf-8")
    assert "run=g9-probe " in log
    assert " supervise g9-1 g9-probe --retries " in log


def test_a_named_run_of_another_entry_is_refused(tmp_path, caplog):
    context = named(tmp_path, {**RUNS, "g9-probe": NamedRun(E1)})
    with caplog.at_level(logging.ERROR):
        assert main(["launch", "g9-1", "e1"], context=context, out=io.StringIO()) == 2
    assert "not a run id of the run 'e1'" in caplog.text
    assert not (tmp_path / "g9-1").exists()
    with pytest.raises(EvaluationError, match="not a run id of the run 'e1'"):
        launch("g9-1", "e1", root=tmp_path, power=lambda: False, start=Starts())


def test_a_named_run_whose_entry_only_starts_with_the_run_ids_is_refused(tmp_path):
    context = named(tmp_path, {**RUNS, "e10": NamedRun(E1)})
    assert main(["launch", "e1-4", "e10", "--dry-run"], context=context, out=io.StringIO()) == 2
    assert not (tmp_path / "e1-4").exists()


@pytest.mark.parametrize("run_id", ["nope", "app-20260101T000000000000Z-bko", "../e1-4"])
def test_an_id_that_is_not_a_run_of_record_exits_two(tmp_path, caplog, run_id):
    with caplog.at_level(logging.ERROR):
        assert main(["launch", run_id], context=named(tmp_path, RUNS), out=io.StringIO()) == 2
    assert "not a run id of record" in caplog.text


@pytest.mark.parametrize(
    "name",
    [f"e6-{fraction}{letter}" for fraction in ("thin", "empty") for letter in "abcdefghij"],
)
def test_each_e6_slice_launches_from_an_e6_run_id(tmp_path, name):
    argv = ["launch", "e6-1", name, "--dry-run"]
    assert main(argv, context=named(tmp_path, RUNS), out=io.StringIO()) == 0
    assert f"run={name} " in (tmp_path / "e6-1" / "launch.log").read_text(encoding="utf-8")


def test_every_named_run_is_an_entry_or_words_after_it():
    assert all(re.fullmatch(r"[a-z]\d+(?:-[a-z]+)*", name) for name in RUNS)
    assert not any(RUN_ID.fullmatch(name) for name in RUNS)


@pytest.mark.parametrize("argv", [["launch", "g2-3"], ["launch", "g2-3", "--dry-run"]])
def test_launch_refuses_g2_whose_only_generator_is_free_before_claiming_a_folder(
    tmp_path, caplog, argv
):
    context = named(tmp_path, RUNS)
    assert main(argv, context=context, out=io.StringIO()) == 2
    assert "the run 'g2' has no paid arm, so it spends nothing to launch" in caplog.text
    assert isinstance(context.start, Starts)
    assert context.start.commands == []
    assert not (tmp_path / "g2-3").exists()


@pytest.mark.parametrize(
    ("run_id", "flags"),
    [("e1-4-resume", {"retries": 0}), ("e1-4-resume", {"rehearse": True}), ("e1-4", {})],
)
def test_a_run_id_supervised_without_resumes_launches(tmp_path, run_id, flags):
    assert launch(run_id, "e1", root=tmp_path, power=lambda: False, start=Starts(), **flags).started


def test_a_run_id_whose_resume_folder_name_is_no_run_id_is_refused_before_its_folder(tmp_path):
    start = Starts()
    message = (
        "e1-4-resume-resume-1 is not a run id, so the supervisor could resume no stop of "
        "'e1-4-resume'; launch under a fresh run id, such as the next free e1-<n>, or pass "
        "--retries 0"
    )
    with pytest.raises(EvaluationError) as refused:
        launch("e1-4-resume", "e1", root=tmp_path, power=lambda: False, start=start)
    assert str(refused.value) == message
    assert not (tmp_path / "e1-4-resume").exists()
    assert start.commands == []
