"""`python -m bko.evaluate supervise`: run a launched run's segments and resume API-caused stops.

`launch` starts this command detached (`bko.evaluate.launch`); `docs/DEVELOPMENT.md` (Evaluation,
Supervision) holds the stop branches it reads and ADR `research/concurrent-call-waves` the decision.
Each segment runs `score` as a child process, under its own `caffeinate -i -s` on macOS so the
machine stays awake while the segment runs even if the supervisor dies, and waits for it. When the
child exits, the segment's call log's `end` line decides: a failed run whose line reads `retryable`,
its stop not a projection or ceiling stop, or a log with no `end` line whose child a signal killed
(a negative exit code: `SIGKILL`, an out-of-memory kill or a sleep), is resumed while resumes
remain; a complete run, any other failure, an interrupt, a log with no `end` line whose child exited
on its own (a crash) and a missing or unreadable log are not. Resume `k` waits `backoff(k)` seconds,
takes the folder `<run_id>-resume-<k>` under launch's own argument, power and folder checks and
claim, and runs `score --resume` on the previous segment's call log there, at half the previous
in-flight (at least 1) when that segment was rate limited. A rehearsal is never resumed. Each
segment's outcome is appended to the run folder's `launch.log`, and the command exits with the last
segment's exit code, or 2 when a resume cannot be claimed.
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
    score_command,
    utc_now,
)
from bko.evaluate.stopping import STOP_FILE, OperatorStop
from bko.generate.errors import ReplayError
from bko.generate.flight import IN_FLIGHT
from bko.generate.replay import UNRESUMABLE, log_ends


logger = logging.getLogger(__name__)

BACKOFF_SECONDS: Final = 60.0
"""The wait before the first resume; each later resume waits twice the one before."""

BACKOFF_CAP_SECONDS: Final = 900.0
"""The longest wait before a resume."""

LINE: Final = (
    "supervise segment=%d folder=%s status=%s stop=%s retryable=%s rate_limited=%s next=%s"
)
"""The line each segment's outcome appends to `launch.log` and logs."""

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
    retryable = status == "failed" and end.get("retryable") is True and stop not in UNRESUMABLE
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
        now: The clock each resume folder's `launch.log` line reads.

    Returns:
        The last segment's exit code, 128 plus the signal's number for a child a signal killed,
        or 2 when a resume's folder cannot be claimed.
    """
    operator = OperatorStop(root / run_id / STOP_FILE)
    with operator.installed():
        return _supervise(
            run_id,
            run,
            root=root,
            data_dir=data_dir,
            rehearse=rehearse,
            resume=resume,
            in_flight=in_flight,
            retries=retries,
            segment=segment,
            wait=operator.wait if sleep is None else sleep,
            power=power,
            platform=platform,
            now=now,
            operator=operator,
        )


def _supervise(  # ruff: ignore[too-many-arguments, too-many-locals]
    run_id: str,
    run: str,
    *,
    root: Path,
    data_dir: Path | None,
    rehearse: bool,
    resume: Path | None,
    in_flight: int | None,
    retries: int,
    segment: Segment,
    wait: Callable[[float], object],
    power: Power,
    platform: str,
    now: Callable[[], datetime],
    operator: OperatorStop,
) -> int:
    """Run the segments as `supervise`, under the operator's stop `operator`."""
    folder = root / run_id / REHEARSAL if rehearse else root / run_id
    record = folder / "launch.log"
    append(record, [f"supervise pid={os.getpid()}"])
    source, flight, k, reading = resume, in_flight, 0, "unchecked"
    while True:
        command = (
            *awake(platform),
            *score_command(
                run, folder, data_dir=data_dir, rehearse=rehearse, resume=source, in_flight=flight
            ),
        )
        if k > 0:
            stamp = now().strftime("%Y-%m-%dT%H:%M:%SZ")
            append(
                folder / "launch.log",
                [
                    f"launched {stamp} run={run} segment={k} of={run_id} resume={source}",
                    f"power={reading}",
                    f"awake={'caffeinate' if awake(platform) else 'none'}",
                    f"command={shlex.join(command)}",
                ],
            )
        started = partial(_started, record, operator, k)
        code = segment(command, folder / "scores.csv", folder / "scores.log", started)
        operator.ended()
        found = outcome(folder / "calls.jsonl", code)
        stopped = operator.stopping and found.status != "complete"
        name = f"{run_id}-resume-{k + 1}"
        resumes = (
            not stopped
            and found.retryable
            and not rehearse
            and k < retries
            and RUN_ID.fullmatch(name) is not None
        )
        if stopped:
            _stopped(record, operator)
        _note(record, k, folder, found, name if resumes else "none")
        if not resumes:
            return exit_code(code)
        k += 1
        if found.rate_limited:
            flight = max(1, (flight or IN_FLIGHT) // 2)
        wait(backoff(k))
        if operator.stopping:
            _stopped(record, operator)
            return exit_code(code)
        source, folder = folder / "calls.jsonl", root / name
        try:
            battery = check_folder(
                name, run, folder, rehearse=False, resume=source, power=power, platform=platform
            )
            claim(folder)
            reading = "unchecked" if battery is None else "mains"
        except EvaluationError as exc:
            reason = " ".join(str(exc).split())
            # The message is the diagnosis; a traceback would read as a crash.
            logger.error("supervise resume=%s refused=%s", name, reason)  # ruff: ignore[error-instead-of-exception]
            append(record, [f"supervise resume={name} refused={reason}"])
            return EvaluationError.exit_code


def _started(record: Path, operator: OperatorStop, k: int, pid: int) -> None:
    """Tell `operator` segment `k`'s child `pid` and append its `child_pid=` line to `record`."""
    operator.started(pid)
    append(record, [f"supervise segment={k} child_pid={pid}"])


def _stopped(record: Path, operator: OperatorStop) -> None:
    """Log the operator's stop and append its `stop=operator` line to `record`."""
    line = f"supervise stop=operator by={operator.by}"
    logger.warning("%s", line)
    append(record, [line])


def _note(record: Path, k: int, folder: Path, found: Outcome, following: str) -> None:
    """Log segment `k`'s outcome and append it to the run's `launch.log` at `record`."""
    values = (
        k,
        folder,
        found.status,
        found.stop or "none",
        str(found.retryable).lower(),
        str(found.rate_limited).lower(),
        following,
    )
    logger.info(LINE, *values)
    append(record, [LINE % values])
