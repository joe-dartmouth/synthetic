"""`python -m bko.evaluate supervise`: the stops it resumes, where, how fast, at which in-flight."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.launch import REHEARSAL
from bko.evaluate.supervise import Outcome, backoff, outcome, run_segment, supervise
from bko.generate.calls import ask
from bko.generate.flight import IN_FLIGHT
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.replay import Resume
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply


type End = Mapping[str, object] | None
"""A scripted segment's `end` line fields; None for a killed child that wrote no `end` line."""

COMPLETE: Final = {"status": "complete", "stop": None, "retryable": False, "rate_limited": False}
IDLE: Final = {"status": "failed", "stop": "idle", "retryable": True, "rate_limited": False}
API: Final = {"status": "failed", "stop": None, "retryable": True, "rate_limited": False}
LIMITED: Final = {**IDLE, "rate_limited": True}
DETERMINISTIC: Final = {"status": "failed", "stop": None, "retryable": False, "rate_limited": False}
CEILING: Final = {"status": "failed", "stop": "ceiling", "retryable": False, "rate_limited": False}
PROJECTION: Final = {**CEILING, "stop": "projection"}
INTERRUPTED: Final = {**DETERMINISTIC, "status": "interrupted"}
KILLED: Final = None
CRASHED: Final = "crashed"
"""A segment that exits on its own, code 1, with no `end` line: a crash, not a kill."""
ABSENT: Final = "absent"
"""A segment that exits before it makes its call log."""

EXIT: Final = {"complete": 0, "failed": 2, "interrupted": 130}
"""The exit code each scripted status returns; a killed child returns -9."""


def _arg(command: Sequence[str], flag: str) -> str | None:
    """The value after `flag` in `command`, or None when it is absent."""
    return command[command.index(flag) + 1] if flag in command else None


class Segments:
    """A fake segment runner: each run writes its scripted `end` line into the command's `--log`.

    Attributes:
        commands: Every command run, in order.
    """

    def __init__(self, *ends: End | str) -> None:
        """Script one segment per end, in order."""
        self.commands: list[tuple[str, ...]] = []
        self._ends = list(ends)

    def __call__(self, command: Sequence[str], out: Path, err: Path) -> int:
        """Write the next end into the call log, beside `out` and `err`; return its exit code."""
        self.commands.append(tuple(command))
        assert out.parent == err.parent
        end = self._ends.pop(0)
        if end == ABSENT:
            return 2
        log = Path(str(_arg(command, "--log")))
        assert log.parent == out.parent
        with CallLog(log, SETTINGS):
            pass
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if end is None or end == CRASHED:
            lines.pop()
        else:
            assert isinstance(end, Mapping)
            lines[-1].update(end)
        log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        if end == CRASHED:
            return 1
        return -9 if end is None else EXIT[str(lines[-1]["status"])]

    @property
    def folders(self) -> list[str]:
        """Each segment's folder name."""
        return [Path(str(_arg(command, "--log"))).parent.name for command in self.commands]

    @property
    def in_flight(self) -> list[str | None]:
        """Each segment's `--in-flight`, or None."""
        return [_arg(command, "--in-flight") for command in self.commands]


def _supervise(
    tmp_path: Path,
    segments: Segments,
    *,
    retries: int = 5,
    in_flight: int | None = None,
    power: bool | None = False,
    rehearse: bool = False,
    run_id: str = "e1-4",
) -> tuple[int, list[float]]:
    """Supervise `run_id` under `tmp_path` through `segments`; return its code and its waits."""
    folder = tmp_path / run_id / REHEARSAL if rehearse else tmp_path / run_id
    folder.mkdir(parents=True)
    waits: list[float] = []
    code = supervise(
        run_id,
        "e1",
        root=tmp_path,
        rehearse=rehearse,
        in_flight=in_flight,
        retries=retries,
        segment=segments,
        sleep=waits.append,
        power=lambda: power,
        platform="darwin",
    )
    return code, waits


def _lines(tmp_path: Path, run_id: str = "e1-4") -> list[str]:
    """The run's `supervise` lines in its `launch.log`."""
    text = (tmp_path / run_id / "launch.log").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.startswith("supervise ")]


def test_supervise_resumes_an_idle_stop_into_resume_1(tmp_path):
    segments = Segments(IDLE, COMPLETE)
    code, _ = _supervise(tmp_path, segments)
    assert code == 0
    assert segments.folders == ["e1-4", "e1-4-resume-1"]
    first, second = segments.commands
    assert _arg(first, "--resume") is None
    assert _arg(second, "--resume") == str(tmp_path / "e1-4" / "calls.jsonl")
    assert second[second.index("score") + 1] == "e1"
    resumed = tmp_path / "e1-4-resume-1"
    assert {"scores.csv", "scores.log", "launch.log"} <= {path.name for path in resumed.iterdir()}
    assert "segment=1 of=e1-4" in (resumed / "launch.log").read_text(encoding="utf-8")
    assert _lines(tmp_path) == [
        (
            f"supervise segment=0 folder={tmp_path / 'e1-4'} status=failed stop=idle "
            "retryable=true rate_limited=false next=e1-4-resume-1"
        ),
        (
            f"supervise segment=1 folder={resumed} status=complete stop=none "
            "retryable=false rate_limited=false next=none"
        ),
    ]


def test_supervise_resumes_an_api_failure(tmp_path):
    segments = Segments(API, COMPLETE)
    assert _supervise(tmp_path, segments)[0] == 0
    assert segments.folders == ["e1-4", "e1-4-resume-1"]


def test_supervise_halves_in_flight_after_rate_limits(tmp_path):
    segments = Segments(LIMITED, {**API, "rate_limited": True}, IDLE, LIMITED, LIMITED, COMPLETE)
    assert _supervise(tmp_path, segments, in_flight=16)[0] == 0
    assert segments.in_flight == ["16", "8", "4", "4", "2", "1"]


def test_supervise_halves_the_default_in_flight_when_none_was_given(tmp_path):
    segments = Segments(IDLE, LIMITED, COMPLETE)
    _supervise(tmp_path, segments)
    assert segments.in_flight == [None, None, str(IN_FLIGHT // 2)]


def test_supervise_resumes_a_killed_segment(tmp_path):
    segments = Segments(KILLED, COMPLETE)
    code, waits = _supervise(tmp_path, segments)
    assert code == 0
    assert segments.folders == ["e1-4", "e1-4-resume-1"]
    assert waits == [60.0]
    assert "status=none stop=none retryable=true" in _lines(tmp_path)[0]


@pytest.mark.parametrize("code", [0, 1, 2])
def test_outcome_reads_a_log_with_no_end_line_as_retryable_only_after_a_signal(tmp_path, code):
    log = tmp_path / "calls.jsonl"
    with CallLog(log, SETTINGS):
        pass
    log.write_text(log.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert outcome(log, code) == Outcome("none", None, retryable=False, rate_limited=False)
    assert outcome(log, -9) == Outcome("none", None, retryable=True, rate_limited=False)


def test_supervise_does_not_resume_a_crash_with_no_end_line(tmp_path):
    segments = Segments(CRASHED)
    assert _supervise(tmp_path, segments) == (1, [])
    assert len(segments.commands) == 1
    assert "status=none stop=none retryable=false" in _lines(tmp_path)[0]


@pytest.mark.parametrize("end", [PROJECTION, CEILING], ids=["projection", "ceiling"])
@pytest.mark.parametrize("marked", [False, True], ids=["as-written", "marked-retryable"])
def test_supervise_does_not_resume_a_projection_or_ceiling_stop(tmp_path, end, marked):
    segments = Segments({**end, "retryable": marked})
    code, waits = _supervise(tmp_path, segments)
    assert (code, waits, len(segments.commands)) == (2, [], 1)
    assert _lines(tmp_path)[0].endswith("retryable=false rate_limited=false next=none")
    assert not (tmp_path / "e1-4-resume-1").exists()


@pytest.mark.parametrize(
    ("end", "code", "status"),
    [(DETERMINISTIC, 2, "failed"), (INTERRUPTED, 130, "interrupted"), (ABSENT, 2, "absent")],
    ids=["deterministic", "interrupted", "no-log"],
)
def test_supervise_does_not_resume_a_deterministic_failure(tmp_path, end, code, status):
    segments = Segments(end)
    assert _supervise(tmp_path, segments) == (code, [])
    assert len(segments.commands) == 1
    assert f"status={status} " in _lines(tmp_path)[0]
    assert _lines(tmp_path)[0].endswith("next=none")


def test_supervise_stops_after_retries_exhausted(tmp_path):
    segments = Segments(IDLE, API, KILLED)
    code, waits = _supervise(tmp_path, segments, retries=2)
    assert code == -9
    assert segments.folders == ["e1-4", "e1-4-resume-1", "e1-4-resume-2"]
    assert len(waits) == 2
    assert _lines(tmp_path)[-1].endswith("retryable=true rate_limited=false next=none")


def test_no_retries_runs_one_segment(tmp_path):
    segments = Segments(IDLE)
    assert _supervise(tmp_path, segments, retries=0) == (2, [])
    assert len(segments.commands) == 1


def test_supervise_backs_off_between_segments(tmp_path):
    segments = Segments(*[IDLE] * 7, COMPLETE)
    _, waits = _supervise(tmp_path, segments, retries=7)
    assert waits == [60.0, 120.0, 240.0, 480.0, 900.0, 900.0, 900.0]
    assert [backoff(k) for k in (1, 4, 5)] == [60.0, 480.0, 900.0]


def test_a_rehearsal_is_never_resumed(tmp_path):
    segments = Segments(IDLE)
    assert _supervise(tmp_path, segments, rehearse=True) == (2, [])
    (command,) = segments.commands
    assert command[-1] == "--rehearse"
    assert segments.folders == [REHEARSAL]


def test_a_run_id_with_no_resume_folder_name_is_not_resumed(tmp_path):
    segments = Segments(IDLE)
    assert _supervise(tmp_path, segments, run_id="e1-4-resume") == (2, [])
    assert _lines(tmp_path, "e1-4-resume")[0].endswith("next=none")


@pytest.mark.parametrize(
    ("power", "taken", "reason"),
    [(True, None, "battery power"), (False, "calls.jsonl", "already holds calls.jsonl")],
    ids=["battery", "taken"],
)
def test_a_resume_launch_refuses_stops_the_run_with_two(tmp_path, power, taken, reason):
    if taken is not None:
        (tmp_path / "e1-4-resume-1").mkdir()
        (tmp_path / "e1-4-resume-1" / taken).write_text("x", encoding="utf-8")
    segments = Segments(IDLE)
    code, waits = _supervise(tmp_path, segments, power=power)
    assert (code, waits, len(segments.commands)) == (2, [60.0], 1)
    refused = (tmp_path / "e1-4" / "launch.log").read_text(encoding="utf-8").splitlines()[-1]
    assert refused.startswith("supervise resume=e1-4-resume-1 refused=")
    assert reason in refused


def test_outcome_reads_an_unreadable_log_as_final(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text("not a log\n", encoding="utf-8")
    assert outcome(log, -9) == Outcome("unreadable", None, retryable=False, rate_limited=False)


def test_run_segment_runs_the_command_to_its_exit_with_its_output_appended(tmp_path):
    out, err = tmp_path / "out", tmp_path / "err"
    out.write_text("kept\n", encoding="utf-8")
    code = "import sys; print('o'); print('e', file=sys.stderr); sys.exit(3)"
    assert run_segment([sys.executable, "-c", code], out, err) == 3
    assert out.read_text(encoding="utf-8") == "kept\no\n"
    assert err.read_text(encoding="utf-8") == "e\n"


def test_the_command_supervises_with_the_contexts_runner_and_sleep(tmp_path):
    segments = Segments(IDLE, COMPLETE)
    waits: list[float] = []
    context = Context(power=lambda: False, runs_dir=tmp_path, segment=segments, sleep=waits.append)
    (tmp_path / "e1-4").mkdir()
    argv = ["supervise", "e1-4", "e1", "--runs-dir", str(tmp_path), "--in-flight", "4"]
    assert main(argv, context=context, out=io.StringIO()) == 0
    assert segments.folders == ["e1-4", "e1-4-resume-1"]
    assert segments.in_flight == ["4", "4"]
    assert waits == [60.0]


def test_a_resumed_log_holds_every_call(tmp_path):
    """Assumption 4: a resume's own call log holds every call, its replayed ones included, so the
    last segment's log replays the whole run."""
    stopped = tmp_path / "stopped.jsonl"
    with CallLog(stopped, SETTINGS) as log:
        for _ in range(2):
            ask(GUIDANCE, 2, send=ScriptedSend(reply([young(1), young(1)])), log=log)
    resume = Resume(stopped, ScriptedSend(reply([young(1), young(1)])))
    resumed = tmp_path / "resumed.jsonl"

    def send(call: Call) -> Reply:
        answer = resume(call)
        assert isinstance(answer, Reply)
        return answer

    with CallLog(resumed, SETTINGS, answered=resume.answered) as log:
        for _ in range(3):
            ask(GUIDANCE, 2, send=send, log=log)
    lines = [json.loads(line) for line in resumed.read_text(encoding="utf-8").splitlines()]
    assert [line["call"] for line in lines if line["kind"] == "request"] == [1, 2, 3]
    assert [line["call"] for line in lines if line["kind"] == "reply"] == [1, 2, 3]
    assert lines[0]["resume"]["calls"] == 2
