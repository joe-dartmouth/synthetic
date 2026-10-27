"""`python -m bko.evaluate launch`: start a paid `score` run detached, awake and on mains power.

ADR `research/e1-evaluation-protocol` records the run's spend rules and `docs/DEVELOPMENT.md`
(Evaluation) the protocol this command carries out. A paid run lost to a laptop sleeping on battery
with a call in flight is what the checks here prevent: the command refuses to start when the machine
reports battery power, starts `score` in its own session with `caffeinate` holding the machine awake
(macOS; elsewhere it records that nothing holds it awake), sends its stdout and stderr to the run
folder's `scores.csv` and `scores.log`, with the call log and the candidates file beside them,
records what it checked and ran in `launch.log`, and prints the command that watches the folder.
A rehearsal (`score --rehearse`, `bko.evaluate.rehearse`) launches the same way into the
`rehearsal` folder inside the run's folder, which leaves the run's own folder free for the run.
It creates the two output files exclusively before it starts
the process, so two launches into one folder cannot both start, and removes them if the process
cannot start or `launch.log` cannot be written, so a failed launch leaves the folder free;
`launch.log` is appended to, so a failed launch's `error=` line survives a retry. A dry run writes
`launch.log` and starts nothing. A resume (`score --resume`) launches into a new folder with the
same checks, and refuses a stopped log inside that folder.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from bko.datasets.run_folders import RUN_ID
from bko.evaluate.errors import EvaluationError


logger = logging.getLogger(__name__)

TAKEN: Final = ("calls.jsonl", "scores.csv", "scores.log", "holdout-candidates.jsonl")
"""The roles a run of record writes; a folder that holds any of them holds a run."""

REHEARSAL: Final = "rehearsal"
"""The folder inside a run's folder its rehearsal writes, so it never takes the run's own roles."""

BATTERY: Final = "'Battery Power'"
"""What `pmset -g batt` prints between quotes when the machine runs on its battery."""

type Power = Callable[[], bool | None]
"""Whether the machine runs on battery: True, False, or None when it cannot be checked."""

type Spawn = Callable[[Sequence[str], Path, Path], int]
"""Start a command detached with its stdout and stderr in two files, and return its process id."""


def on_battery() -> bool | None:
    """Whether the machine reports battery power: `pmset -g batt` on macOS, None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("pmset unreadable error=%s", type(exc).__name__)
        return None
    if result.returncode != 0:
        logger.warning("pmset unreadable returncode=%d", result.returncode)
        return None
    return BATTERY in result.stdout


def spawn(command: Sequence[str], out: Path, err: Path) -> int:
    """Start `command` in its own session with stdin closed and stdout and stderr in files."""
    with out.open("ab") as stdout, err.open("ab") as stderr:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    return process.pid


@dataclass(frozen=True, slots=True)
class Launch:
    """What `launch` did.

    Attributes:
        folder: The run folder.
        command: The command it started, or would start in a dry run.
        watch: The command that watches the folder.
        started: Whether the process started.
        pid: Its process id, or None in a dry run.
    """

    folder: Path
    command: tuple[str, ...]
    watch: str
    started: bool
    pid: int | None


def launch(  # ruff: ignore[too-many-arguments]
    run_id: str,
    run: str,
    *,
    root: Path,
    data_dir: Path | None = None,
    dry_run: bool = False,
    rehearse: bool = False,
    resume: Path | None = None,
    power: Power = on_battery,
    start: Spawn = spawn,
    platform: str = sys.platform,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Launch:
    """Start `score` for the named `run` into `root/<run_id>`, or only write the folder's roles.

    Args:
        run_id: The run's id, such as `e1-4`.
        run: The named run `score` takes: its entry, `<entry>` or `<entry>-<word>`, is the run
            id's.
        root: The run folders' root.
        data_dir: The data root `score` reads, or its default.
        dry_run: Write the folder's roles and the log, and start nothing.
        rehearse: Rehearse the run on canned replies, into `root/<run_id>/rehearsal`.
        resume: The call log of a stopped run to resume, or None for a new run.
        power: Whether the machine runs on battery.
        start: Starts the command detached.
        platform: `sys.platform`; `caffeinate` holds the machine awake on `darwin` only.
        now: The clock the log's line reads.

    Raises:
        EvaluationError: If `run_id` is not a run id of record of `run`'s entry, `data_dir`
            reads as an option, `rehearse` is given with `resume`, `resume` is not a file or lies
            inside the folder, the machine reports battery power, or on macOS its power cannot
            be read, the folder already holds the call log, the scores, their log or the
            candidates file, the output files cannot be claimed, or the process cannot start.
        OSError: If `launch.log` cannot be written; the claimed files are removed first.
    """
    folder = root / run_id / REHEARSAL if rehearse else root / run_id
    _refuse_arguments(run_id, run, folder, data_dir=data_dir, rehearse=rehearse, resume=resume)
    battery = _mains(power, platform)
    taken = [name for name in TAKEN if (folder / name).exists()]
    if taken:
        msg = f"{folder} already holds {', '.join(taken)}; a run of record gets a new folder"
        raise EvaluationError(msg)
    awake = ["caffeinate", "-i", "-s"] if platform == "darwin" else []
    command = (
        *awake,
        sys.executable,
        "-m",
        "bko.evaluate",
        "score",
        run,
        *(["--resume", str(resume.absolute())] if resume is not None else []),
        "--log",
        str(folder / "calls.jsonl"),
        "--candidates",
        str(folder / "holdout-candidates.jsonl"),
        *(["--data-dir", str(data_dir)] if data_dir is not None else []),
        *(["--rehearse"] if rehearse else []),
    )
    folder.mkdir(parents=True, exist_ok=True)
    out, err = folder / "scores.csv", folder / "scores.log"
    reading = "unchecked" if battery is None else "mains"
    lines = [
        (
            f"launched {now().strftime('%Y-%m-%dT%H:%M:%SZ')} run={run} dry_run={dry_run} "
            f"rehearse={str(rehearse).lower()} resume={resume.absolute() if resume else 'none'}"
        ),
        f"power={reading}",
        f"awake={'caffeinate' if awake else 'none'}",
        f"command={shlex.join(command)}",
    ]
    claimed = () if dry_run else _claim(out, err)
    record = folder / "launch.log"
    try:
        _append(record, lines)
    except BaseException:
        _release(claimed)
        raise
    pid = None if dry_run else _start(start, command, claimed, record)
    _append(record, [f"pid={pid}"])
    watch = f"tail -f {shlex.quote(str(err))}"
    logger.info("launch folder=%s power=%s dry_run=%s pid=%s", folder, reading, dry_run, pid)
    return Launch(folder, command, watch, started=not dry_run, pid=pid)


def _refuse_arguments(
    run_id: str,
    run: str,
    folder: Path,
    *,
    data_dir: Path | None,
    rehearse: bool,
    resume: Path | None,
) -> None:
    """Refuse the arguments `launch` names before it reads the power or touches the folder.

    Raises:
        EvaluationError: As `launch`, for each check on its arguments.
    """
    match = RUN_ID.fullmatch(run_id)
    if match is None or match["entry"] is None:
        msg = f"{run_id!r} is not a run id of record, such as e1-4"
        raise EvaluationError(msg)
    if run.partition("-")[0] != match["entry"]:
        msg = f"{run_id!r} is not a run id of the run {run!r}, such as {run.partition('-')[0]}-4"
        raise EvaluationError(msg)
    if rehearse and resume is not None:
        msg = "a rehearsal answers every call with canned replies, so it resumes nothing"
        raise EvaluationError(msg)
    if data_dir is not None and str(data_dir).startswith("-"):
        msg = f"the data root {str(data_dir)!r} reads as an option; name it ./{data_dir}"
        raise EvaluationError(msg)
    if resume is not None and not resume.is_file():
        msg = f"the resume source {resume} is not a file"
        raise EvaluationError(msg)
    if resume is not None and resume.resolve().is_relative_to(folder.resolve()):
        msg = f"the resume source {resume} lies inside {folder}, the folder the resume writes"
        raise EvaluationError(msg)


def _mains(power: Power, platform: str) -> bool | None:
    """The power reading, refused on battery and, on macOS, when it cannot be read.

    Raises:
        EvaluationError: As `launch`, for the power checks.
    """
    battery = power()
    if battery:
        msg = (
            "the machine reports battery power; a paid run lost to sleep is lost money, so plug in"
        )
        raise EvaluationError(msg)
    if battery is None and platform == "darwin":
        msg = "pmset could not report the machine's power, so mains power cannot be confirmed"
        raise EvaluationError(msg)
    return battery


def _start(start: Spawn, command: Sequence[str], claimed: Sequence[Path], record: Path) -> int:
    """Start `command` with its output in the claimed files; release them if it fails.

    Raises:
        EvaluationError: If the process cannot start, an `OSError`; any other failure propagates.
            Either way the launch log gets its `error=` line.
    """
    out, err = claimed
    try:
        return start(command, out, err)
    except BaseException as exc:
        _release(claimed)
        with contextlib.suppress(OSError):
            _append(record, [f"error={type(exc).__name__}"])
        if not isinstance(exc, OSError):
            raise
        msg = f"{command[0]} could not start: {exc}"
        raise EvaluationError(msg) from exc


def _append(record: Path, lines: Sequence[str]) -> None:
    """Append `lines` to the launch log at `record`, so earlier launches' lines survive."""
    with record.open("a", encoding="utf-8") as appending:
        appending.write("".join(f"{line}\n" for line in lines))


def _release(claimed: Sequence[Path]) -> None:
    """Remove the files a launch claimed, so the folder is free again."""
    for path in claimed:
        path.unlink(missing_ok=True)


def _claim(out: Path, err: Path) -> tuple[Path, Path]:
    """Create `out` and `err` exclusively, so one launch alone owns the folder.

    Returns:
        The two files, for `_release`.

    Raises:
        EvaluationError: If another launch created either file first, or either cannot be
            created; any file made is removed first.
    """
    made: list[Path] = []
    try:
        for path in (out, err):
            path.open("x").close()
            made.append(path)
    except FileExistsError:
        _release(made)
        msg = f"{out.parent} already holds a run's output; another launch started first"
        raise EvaluationError(msg) from None
    except OSError as exc:
        _release(made)
        msg = f"{out.parent} could not claim a run's output: {exc}"
        raise EvaluationError(msg) from exc
    return out, err
