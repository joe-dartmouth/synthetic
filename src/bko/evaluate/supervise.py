"""`python -m bko.evaluate supervise`: run a launched run's segments and resume API-caused stops.

`launch` starts this command detached (`bko.evaluate.launch`); it is internal to `launch`.
`docs/DEVELOPMENT.md` (Evaluation, Supervision) holds the stop branches it reads, every line it
writes and the operator's stop, and ADR `research/concurrent-call-waves` the decision. Each segment
runs `score` as a child process in its own process group, under its own `caffeinate -i -s` on macOS
so the machine stays awake while the segment runs even if the supervisor dies, and waits for it.
When the child exits, the segment's call log's `end` line and the child's exit code decide, and the
segment's line in `launch.log` names the decision under `reason`, one of `REASONS`: a failed run
whose line reads `retryable`, its stop not a projection or ceiling stop, or a log with no `end` line
whose child a signal killed (a negative exit code: `SIGKILL`, an out-of-memory kill or a sleep), is
resumed while resumes remain; a complete run, any other failure, an interrupt, a log with no `end`
line whose child exited on its own (a crash), a missing or unreadable log and an operator's stop
(`bko.evaluate.stopping`) are not. Resume `k` waits `backoff(k)` seconds, takes the folder
`<run_id>-resume-<k>` under launch's own argument, power and folder checks and claim, and runs
`score --resume` on the previous segment's call log there, at half the previous in-flight (at
least 1) when that segment was rate limited. A rehearsal is never resumed, and neither is a run id
whose `-resume-<k>` folder name is not a `RUN_ID`. Every line goes to the run folder's `launch.log`
behind a UTC time, and to the log; the last is `supervise done`, with the command's exit code: the
last segment's, 128 plus the signal's number for a child a signal killed, or 2 when a resume
cannot be claimed.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Final

from bko.datasets.run_folders import RUN_ID
from bko.evaluate.errors import EvaluationError
from bko.evaluate.launch import (
    REHEARSAL,
    Power,
    append,
    awake,
    check_folder,
    claim,
    on_battery,
    release,
    resumable,
    score_command,
    utc_now,
)
from bko.evaluate.stopping import STOP_FILE, OperatorStop
from bko.generate.errors import ReplayError
from bko.generate.flight import IN_FLIGHT
from bko.generate.log import SPEND_STOPS
from bko.generate.messages import Settings
from bko.generate.resume_guard import log_ends


logger = logging.getLogger(__name__)

BACKOFF_SECONDS: Final = 60.0
"""The wait before the first resume; each later resume waits twice the one before."""

BACKOFF_CAP_SECONDS: Final = 900.0
"""The longest wait before a resume."""

STAMP: Final = "%Y-%m-%dT%H:%M:%SZ"
"""The UTC time format that starts every `launch.log` line the supervisor writes."""

REASONS: Final = (
    "complete",
    "stop",
    "retryable",
    "exhausted",
    "unresumable",
    "crash",
    "rehearsal",
    "name",
    "operator",
    "refused",
    "error",
)
"""Why a segment's line or the `done` line ends as it does: the run completed; the projection or
the ceiling stopped it; a resume is scheduled; the resumes ran out; the failure is not the API's,
or the log is missing or unreadable; the child exited on its own with no `end` line; a rehearsal is
never resumed; the run id's resume folder name is not a `RUN_ID`; the operator stopped the run; a
resume's checks refused it, or a segment could not start (these two on the `done` line alone)."""

LINE: Final = (
    "supervise segment=%d folder=%s code=%d status=%s stop=%s retryable=%s rate_limited=%s "
    "reason=%s"
)
"""The line each segment's outcome appends to `launch.log` and logs."""

SCHEDULED: Final = " backoff_s=%d"
"""What a segment's line adds when a resume is scheduled: the backoff before it."""

NEXT: Final = "supervise next=%s in_flight=%s watch=%s"
"""The line a claimed resume appends: its folder, its in-flight and the command that watches it."""

DONE: Final = "supervise done segments=%d last=%s outcome=%s exit=%d"
"""The supervisor's last line: segments run, the last one's folder, the reason, the exit code."""

type Segment = Callable[[Sequence[str], Path, Path, Callable[[int], None]], int]
"""Run a command to its exit, its stdout and stderr appended to two files, telling the callback its
process id once it starts; return its exit code."""


def run_segment(
    command: Sequence[str], out: Path, err: Path, started: Callable[[int], None]
) -> int:
    """Run `command` in its own process group, stdin closed, stdout and stderr appended to files.

    It waits for the child with no timeout, so a stopped child finishes its call log's `end`
    line; a signal to the caller's process group does not reach it.

    Args:
        command: The command.
        out: The file its stdout is appended to.
        err: The file its stderr is appended to.
        started: Told the child's process id once it starts.

    Returns:
        Its exit code, negative for a signal that killed it.

    Raises:
        OSError: If the command cannot start.
    """
    with out.open("ab") as stdout, err.open("ab") as stderr:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(command), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, process_group=0
        )
    started(process.pid)
    return process.wait()


def exit_code(code: int) -> int:
    """A child's exit `code` as a process exit status: a signal's negative code becomes 128 + it."""
    return 128 - code if code < 0 else code


def backoff(resume: int) -> float:
    """The seconds resume `resume` (from 1) waits: 60 doubling each time, capped at 900."""
    return min(BACKOFF_SECONDS * 2 ** (resume - 1), BACKOFF_CAP_SECONDS)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a finished segment's call log says.

    Attributes:
        status: The `end` line's status; `none` for a log with no `end` line, `absent` for no
            log, `unreadable` for a log whose first line is not a call log's.
        stop: The stop rule the `end` line names, or None.
        retryable: Whether the API caused the stop, or the log has no `end` line and a signal
            killed the child: the segment is resumed while resumes remain.
        rate_limited: Whether any retry in the segment met a 429.
    """

    status: str
    stop: str | None
    retryable: bool
    rate_limited: bool


def outcome(log: Path, code: int) -> Outcome:
    """Read a finished segment's call log at `log`; its child process, which exited `code`, is gone.

    A log with no `end` line is retryable only when `code` is negative, a signal's: `score` turns
    `SIGTERM`, `SIGINT` and `SIGHUP` into an `interrupted` end line, so a signal that leaves none
    is a kill, and a child that exited on its own with none crashed.
    """
    if not log.is_file():
        return Outcome("absent", None, retryable=False, rate_limited=False)
    try:
        _, end = log_ends(log)
    except ReplayError, OSError:
        return Outcome("unreadable", None, retryable=False, rate_limited=False)
    if end is None:
        return Outcome("none", None, retryable=code < 0, rate_limited=False)
    stop = end.get("stop")
    status = str(end.get("status"))
    retryable = status == "failed" and end.get("retryable") is True and stop not in SPEND_STOPS
    return Outcome(
        status,
        stop if isinstance(stop, str) else None,
        retryable=retryable,
        rate_limited=end.get("rate_limited") is True,
    )


def supervise(  # ruff: ignore[too-many-arguments]
    run_id: str,
    run: str,
    *,
    root: Path,
    data_dir: Path | None = None,
    rehearse: bool = False,
    resume: Path | None = None,
    in_flight: int | None = None,
    retries: int,
    segment: Segment = run_segment,
    sleep: Callable[[float], object] | None = None,
    power: Power = on_battery,
    platform: str = sys.platform,
    now: Callable[[], datetime] = utc_now,
    settings: Settings,
) -> int:
    """Run `score` for the named `run` into `root/<run_id>`, resuming API-caused stops.

    The first segment's folder is the one `launch` claimed; it writes into it and appends to its
    `scores.csv` and `scores.log`.

    Args:
        run_id: The run's id, such as `e1-4`.
        run: The named run `score` takes.
        root: The run folders' root.
        data_dir: The data root `score` reads, or None for its default.
        rehearse: Rehearse the run on canned replies, into `root/<run_id>/rehearsal`; a rehearsal
            is never resumed.
        resume: The stopped call log the first segment resumes, or None for a new run.
        in_flight: The calls `score` keeps in flight, or None for its default.
        retries: The resumes made at most; 0 runs one segment.
        segment: Runs one segment's command to its exit.
        sleep: Waits the backoff before each resume; None waits until the backoff ends or the
            operator stops the run.
        power: Whether the machine runs on battery, read before each resume.
        platform: `sys.platform`, as `launch` reads it; each segment runs under `caffeinate`
            on `darwin`.
        now: The clock every `launch.log` line reads.
        settings: The named run's generator settings; a resume refuses a stopped log made with
            others before it claims its folder.

    Returns:
        The last segment's exit code, 128 plus the signal's number for a child a signal killed,
        or 2 when a resume's folder cannot be claimed.
    """
    operator = OperatorStop(root / run_id / STOP_FILE)
    plan = _Plan(
        run_id,
        run,
        root,
        data_dir,
        rehearse=rehearse,
        retries=retries,
        segment=segment,
        wait=operator.wait if sleep is None else sleep,
        power=power,
        platform=platform,
        now=now,
        operator=operator,
        settings=settings,
    )
    with operator.installed():
        return _supervise(plan, resume, in_flight)


@dataclass(frozen=True, slots=True)
class _Plan:
    """What every segment of one supervised run shares: `supervise`'s arguments.

    Attributes:
        run_id: The run's id.
        run: The named run.
        root: The run folders' root.
        data_dir: The data root, or None.
        rehearse: Whether the run is a rehearsal.
        retries: The resumes made at most.
        segment: Runs one segment.
        wait: Waits a backoff.
        power: Reads the power.
        platform: `sys.platform`.
        now: The clock.
        operator: The operator's stop.
        settings: The named run's generator settings.
    """

    run_id: str
    run: str
    root: Path
    data_dir: Path | None
    rehearse: bool
    retries: int
    segment: Segment
    wait: Callable[[float], object]
    power: Power
    platform: str
    now: Callable[[], datetime]
    operator: OperatorStop
    settings: Settings

    @property
    def first(self) -> Path:
        """The first segment's folder, which `launch` claimed; its `launch.log` is the record."""
        return self.root / self.run_id / REHEARSAL if self.rehearse else self.root / self.run_id

    def write(self, line: str, *, level: int = logging.INFO, record: Path | None = None) -> None:
        """Log `line` at `level` and append it, behind the time, to `record` or the run's log."""
        logger.log(level, "%s", line)
        append(record or self.first / "launch.log", [f"{self.now().strftime(STAMP)} {line}"])

    def command(self, folder: Path, source: Path | None, flight: int | None) -> tuple[str, ...]:
        """The segment's command into `folder`, resuming `source` at `flight`, kept awake."""
        return (
            *awake(self.platform),
            *score_command(
                self.run,
                folder,
                data_dir=self.data_dir,
                rehearse=self.rehearse,
                resume=source,
                in_flight=flight,
            ),
        )

    def reason(self, k: int, found: Outcome) -> str:
        """Why segment `k`, which ended as `found`, is resumed or not: one of `REASONS`."""
        checks = (
            (found.status == "complete", "complete"),
            (self.operator.stopping or found.status == "interrupted", "operator"),
            (found.stop in SPEND_STOPS, "stop"),
            (found.status == "none" and not found.retryable, "crash"),
            (not found.retryable, "unresumable"),
            (self.rehearse, "rehearsal"),
            (RUN_ID.fullmatch(f"{self.run_id}-resume-{k + 1}") is None, "name"),
            (k >= self.retries, "exhausted"),
        )
        return next((why for hit, why in checks if hit), "retryable")

    def done(self, k: int, folder: Path, why: str, code: int) -> int:
        """Write the `done` line after segment `k` in `folder`, and return the exit `code`."""
        self.write(DONE % (k + 1, folder.name, why, code))
        return code

    def stopped(self) -> None:
        """Write the operator's `stop=operator` line, a WARNING."""
        self.write(f"supervise stop=operator by={self.operator.by}", level=logging.WARNING)


def _supervise(plan: _Plan, resume: Path | None, in_flight: int | None) -> int:
    """Run `plan`'s segments, the first resuming `resume` at `in_flight`, as `supervise`."""
    folder, source, flight, k, reading = plan.first, resume, in_flight, 0, "unchecked"
    claimed: tuple[Path, ...] = ()
    plan.write(f"supervise pid={os.getpid()}")
    while True:
        command = plan.command(folder, source, flight)
        try:
            if k > 0:
                _opened(plan, folder, command, f"segment={k} resume={source}", reading)
            code = plan.segment(
                command, folder / "scores.csv", folder / "scores.log", partial(_started, plan, k)
            )
        except OSError as exc:
            return _failed(plan, k, folder, exc, claimed)
        plan.operator.ended()
        found = outcome(folder / "calls.jsonl", code)
        why = plan.reason(k, found)
        if plan.operator.stopping and found.status != "complete":
            plan.stopped()
        line = LINE % (k, folder, code, *_fields(found), why)
        plan.write(line + (SCHEDULED % backoff(k + 1) if why == "retryable" else ""))
        if why != "retryable":
            return plan.done(k, folder, why, exit_code(code))
        if found.rate_limited:
            flight = max(1, (flight or IN_FLIGHT) // 2)
        plan.wait(backoff(k + 1))
        if plan.operator.stopping:
            plan.stopped()
            return plan.done(k, folder, "operator", exit_code(code))
        name = f"{plan.run_id}-resume-{k + 1}"
        try:
            reading, claimed = _claim(plan, name, folder / "calls.jsonl")
        except EvaluationError as exc:
            reason = " ".join(str(exc).split())
            plan.write(f"supervise resume={name} refused={reason}", level=logging.ERROR)
            return plan.done(k, folder, "refused", EvaluationError.exit_code)
        source, folder, k = folder / "calls.jsonl", plan.root / name, k + 1
        watch = f"tail -f {shlex.quote(str(folder / 'scores.log'))}"
        plan.write(NEXT % (name, flight or "default", watch))


def _claim(plan: _Plan, name: str, source: Path) -> tuple[str, tuple[Path, Path]]:
    """Check `source` and the resume folder `name`, then claim it.

    Returns:
        Its power reading, and the files claimed, for `release`.

    Nothing is claimed until every check passed, so a refused resume leaves no folder claimed.

    Raises:
        EvaluationError: As `bko.evaluate.launch.check_folder`, `resumable` and `claim`.
    """
    folder = plan.root / name
    battery = check_folder(
        name,
        plan.run,
        folder,
        rehearse=False,
        resume=source,
        power=plan.power,
        platform=plan.platform,
    )
    resumable(source, plan.settings)
    return "unchecked" if battery is None else "mains", claim(folder)


def _failed(plan: _Plan, k: int, folder: Path, exc: OSError, claimed: Sequence[Path]) -> int:
    """Release segment `k`'s `claimed` files, which it never wrote, and write why it failed.

    A `launch.log` that cannot take the lines is logged instead.

    Returns:
        2, the exit code of a command that could not run.
    """
    release(claimed)
    reason = " ".join(f"{type(exc).__name__}: {exc}".split())
    try:
        plan.write(f"supervise error={reason}", level=logging.ERROR)
        return plan.done(k, folder, "error", EvaluationError.exit_code)
    except OSError as unwritten:
        logger.warning(
            "supervise error unrecorded error=%s: %s", type(unwritten).__name__, unwritten
        )
    return EvaluationError.exit_code


def _opened(plan: _Plan, folder: Path, command: Sequence[str], what: str, reading: str) -> None:
    """Append a resume's `launched`, `power=`, `awake=` and `command=` lines to its `launch.log`."""
    segment, resume = what.split(" ", 1)
    stamp = plan.now().strftime(STAMP)
    append(
        folder / "launch.log",
        [
            f"launched {stamp} run={plan.run} {segment} of={plan.run_id} {resume}",
            f"power={reading}",
            f"awake={'caffeinate' if awake(plan.platform) else 'none'}",
            f"command={shlex.join(command)}",
        ],
    )


def _started(plan: _Plan, k: int, pid: int) -> None:
    """Tell the operator's stop segment `k`'s child `pid` and write its `child_pid=` line.

    The child runs already, so a line that cannot be written is logged as a WARNING instead.
    """
    plan.operator.started(pid)
    try:
        plan.write(f"supervise segment={k} child_pid={pid}")
    except OSError as exc:
        logger.warning(
            "supervise child_pid=%d unrecorded error=%s: %s", pid, type(exc).__name__, exc
        )


def _fields(found: Outcome) -> tuple[str, str, str, str]:
    """`found`'s status, stop, `retryable` and `rate_limited` as a segment's line writes them."""
    return (
        found.status,
        found.stop or "none",
        str(found.retryable).lower(),
        str(found.rate_limited).lower(),
    )
