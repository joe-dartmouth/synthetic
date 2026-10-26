"""`python -m bko.evaluate launch`: start a paid `score` run detached, awake and on mains power.

ADR `research/e1-evaluation-protocol` records the run's spend rules and `docs/DEVELOPMENT.md`
(Evaluation) the protocol this command carries out. A paid run lost to a laptop sleeping on battery
with a call in flight is what the checks here prevent: the command refuses to start when the machine
reports battery power, starts `score` in its own session with `caffeinate` holding the machine awake
(macOS; elsewhere it records that nothing holds it awake), sends its stdout and stderr to the run
folder's `scores.csv` and `scores.log`, records what it checked and ran in `launch.log`, and prints
the command that watches the folder. A dry run does everything but start the process.
"""

from __future__ import annotations

import logging
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
    except OSError, subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
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


def launch(
    run_id: str,
    run: str,
    *,
    root: Path,
    data_dir: Path | None = None,
    dry_run: bool = False,
    power: Power = on_battery,
    start: Spawn = spawn,
    platform: str = sys.platform,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Launch:
    """Start `score` for the named `run` into `root/<run_id>`, or only write the folder's roles.

    Args:
        run_id: The run's id, such as `e1-4`.
        run: The named run `score` takes.
        root: The run folders' root.
        data_dir: The data root `score` reads, or its default.
        dry_run: Write the folder's roles and the log, and start nothing.
        power: Whether the machine runs on battery.
        start: Starts the command detached.
        platform: `sys.platform`; `caffeinate` holds the machine awake on `darwin` only.
        now: The clock the log's line reads.

    Raises:
        EvaluationError: If `run_id` is not a run id of record, the machine reports battery power,
            or the folder already holds the call log, the scores or their log.
    """
    match = RUN_ID.fullmatch(run_id)
    if match is None or match["entry"] is None:
        msg = f"{run_id!r} is not a run id of record, such as e1-4"
        raise EvaluationError(msg)
    battery = power()
    if battery:
        msg = (
            "the machine reports battery power; a paid run lost to sleep is lost money, so plug in"
        )
        raise EvaluationError(msg)
    folder = root / run_id
    taken = [
        name for name in ("calls.jsonl", "scores.csv", "scores.log") if (folder / name).exists()
    ]
    if taken:
        msg = f"{folder} already holds {', '.join(taken)}; a run of record gets a new folder"
        raise EvaluationError(msg)
    awake = ["caffeinate", "-i"] if platform == "darwin" else []
    command = (
        *awake,
        sys.executable,
        "-m",
        "bko.evaluate",
        "score",
        run,
        "--log",
        str(folder / "calls.jsonl"),
        *(["--data-dir", str(data_dir)] if data_dir is not None else []),
    )
    folder.mkdir(parents=True, exist_ok=True)
    out, err = folder / "scores.csv", folder / "scores.log"
    out.touch()
    err.touch()
    reading = "unchecked" if battery is None else "mains"
    pid = None if dry_run else start(command, out, err)
    lines = [
        f"launched {now().strftime('%Y-%m-%dT%H:%M:%SZ')} run={run} dry_run={dry_run}",
        f"power={reading}",
        f"awake={'caffeinate' if awake else 'none'}",
        f"pid={pid}",
        f"command={' '.join(command)}",
    ]
    (folder / "launch.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    watch = f"tail -f {err}"
    logger.info("launch folder=%s power=%s dry_run=%s pid=%s", folder, reading, dry_run, pid)
    return Launch(folder, command, watch, started=not dry_run, pid=pid)
