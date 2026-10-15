"""Generate a Synthea population with the pinned jar, every output-changing parameter pinned.

ADR `research/synthea-ablation-design` records why each parameter is pinned. A population
directory `<data root>/synthea/<key>/` appears only through a rename after Synthea exits 0 and its
CSVs exist, so one that exists is complete. `run.json` in it records the parameters, the Java
version and each CSV's sha256 and row count, the identity `docs/EVIDENCE.md` asks an experiment to
cite. A failed run leaves its log at `<data root>/synthea/<key>.log`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

from bko.datasets.errors import ContractError, DatasetError, UnavailableError
from bko.datasets.fetch import UrlOpener, ensure, open_https, staging_beside
from bko.datasets.sources import SYNTHEA_JAR, Source


logger = logging.getLogger(__name__)

GENERATION_TIMEOUT_SECONDS: Final = 3600.0
"""How long one generation may run before it is killed."""

TABLES: Final = ("patients.csv", "conditions.csv")
"""The CSVs Synthea writes, and the only ones loaded."""

RUN_RECORD_NAME: Final = "run.json"
"""The file in a population directory that records how it was generated."""

LOG_NAME: Final = "synthea.log"
"""Synthea's combined output, kept in the population directory."""

JVM_OPTIONS: Final = (
    "-Duser.timezone=UTC",
    "-Duser.language=en",
    "-Duser.country=US",
    "-Dfile.encoding=UTF-8",
)
"""JVM settings that would otherwise follow the machine: time zone, locale and text encoding."""

CHILD_ENVIRONMENT: Final[Mapping[str, str]] = {
    "TZ": "UTC",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
"""Environment variables every Java process gets, beside `PATH` and `HOME`; no other is passed."""

CONFIG: Final = (
    "--exporter.csv.export=true",
    f"--exporter.csv.included_files={','.join(TABLES)}",
    "--exporter.fhir.export=false",
    "--exporter.fhir.transaction_bundle=false",
    "--exporter.hospital.fhir.export=false",
    "--exporter.practitioner.fhir.export=false",
    "--exporter.years_of_history=0",
    "--generate.thread_pool_size=1",
    "--generate.log_patients.detail=none",
)
"""Settings every run passes: CSV only, the two tables, the full history, one thread."""

STATE: Final = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*")
"""A US state name as Synthea takes it, such as `Massachusetts` or `New York`."""

type Runner = Callable[[Sequence[str], Path], int]
"""Run Synthea's argv in the given directory, writing its output to `LOG_NAME` there, and return
its exit status."""


@dataclass(frozen=True, slots=True)
class SyntheaRun:
    """The parameters that, with the pinned jar and `CONFIG`, determine a population.

    Attributes:
        seed: Synthea's `-s`.
        clinician_seed: Synthea's `-cs`.
        population: Synthea's `-p`, the patients alive at the end date.
        snapshot_date: Both `-r` (ages are anchored to it) and `-e` (the simulation ends on it).
        state: The US state Synthea draws the population from.
    """

    seed: int
    clinician_seed: int
    population: int
    snapshot_date: date
    state: str = "Massachusetts"

    def __post_init__(self) -> None:
        """Refuse a population under 1 or a state that is not a plain state name.

        Raises:
            ValueError: If either is out of range.
        """
        if self.population < 1:
            msg = f"population must be at least 1, not {self.population}"
            raise ValueError(msg)
        if not STATE.fullmatch(self.state):
            msg = f"state must be a US state name such as 'New York', not {self.state!r}"
            raise ValueError(msg)


DEFAULT_RUN: Final = SyntheaRun(
    seed=1, clinician_seed=1, population=2_000, snapshot_date=date(2026, 1, 1)
)
"""The population the command generates, sized by the pilot ADR `research/synthea-ablation-design`
records."""


def population_dir(run: SyntheaRun, data_dir: Path, jar_source: Source = SYNTHEA_JAR) -> Path:
    """Return the directory that holds the population `run` generates with `jar_source`."""
    key = (
        f"{jar_source.sha256[:16]}-s{run.seed}-cs{run.clinician_seed}-p{run.population}"
        f"-d{run.snapshot_date:%Y%m%d}-{run.state.lower().replace(' ', '-')}"
    )
    return data_dir / "synthea" / key


def synthea_argv(run: SyntheaRun, *, java: Path, jar: Path, out_dir: Path) -> list[str]:
    """Return the command that generates `run` into `out_dir`, every path absolute."""
    stamp = f"{run.snapshot_date:%Y%m%d}"
    return [
        str(java.resolve()),
        *JVM_OPTIONS,
        "-jar",
        str(jar.resolve()),
        "-s",
        str(run.seed),
        "-cs",
        str(run.clinician_seed),
        "-r",
        stamp,
        "-e",
        stamp,
        "-p",
        str(run.population),
        f"--exporter.baseDirectory={out_dir.resolve()}/",
        *CONFIG,
        run.state,
    ]


def find_java() -> Path:
    """Return `$JAVA_HOME/bin/java` when `JAVA_HOME` is set, else `java` on `PATH`, resolved.

    A `JAVA_HOME` that holds no java is an error, not a reason to fall back to `PATH`.

    Raises:
        UnavailableError: If the chosen java is absent or not executable.
    """
    home = os.environ.get("JAVA_HOME")
    if home:
        candidate: Path | None = Path(home).resolve() / "bin" / "java"
        where = f"JAVA_HOME ({candidate})"
    else:
        found = shutil.which("java")
        candidate = Path(found).resolve() if found else None
        where = "PATH"
    if candidate is None or not candidate.is_file() or not os.access(candidate, os.X_OK):
        msg = f"no runnable java on {where}; Synthea v4.0.0 needs JDK 17 or newer"
        raise UnavailableError(msg)
    return candidate


def probe_version(java: Path) -> str:
    """Return the version line `java -version` prints to stderr, run as Synthea's java runs.

    Raises:
        UnavailableError: If it cannot run, times out, or exits non-zero, as the macOS stub does
            without a JDK.
    """
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [str(java), *JVM_OPTIONS, "-version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=child_environment(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        msg = f"{java} -version failed: {exc}"
        raise UnavailableError(msg) from exc
    if result.returncode != 0:
        msg = f"{java} -version exited {result.returncode}: {result.stderr.strip()}"
        raise UnavailableError(msg)
    lines = result.stderr.splitlines()
    return next((line for line in lines if " version " in line), lines[0] if lines else "")


def default_java() -> tuple[Path, str]:
    """Return the Java to run Synthea with and its version line."""
    java = find_java()
    return java, probe_version(java)


def child_environment() -> dict[str, str]:
    """Return the environment a Java process gets: `PATH`, `HOME` and `CHILD_ENVIRONMENT`."""
    kept = {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}
    return kept | dict(CHILD_ENVIRONMENT)


def run_java(argv: Sequence[str], cwd: Path, *, timeout: float = GENERATION_TIMEOUT_SECONDS) -> int:
    """Run `argv` in `cwd` with its output in `cwd/synthea.log`, and return its exit status.

    Raises:
        subprocess.TimeoutExpired: If it runs past `timeout`; it is killed first.
    """
    with (cwd / LOG_NAME).open("wb") as log:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            cwd=cwd,
            env=child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return result.returncode


def check_population(directory: Path, run: SyntheaRun, jar_source: Source = SYNTHEA_JAR) -> None:
    """Check that `directory` holds the population `run` generates, against its `run.json`.

    The record must name exactly `TABLES`, match `run`, the jar and the pinned settings, and each
    CSV must match its recorded sha256. This detects a changed or partial population; it does not
    authenticate one, since the record sits beside the tables.

    Raises:
        DatasetError: If the record or a CSV is missing or unreadable, or anything differs.
    """
    try:
        record, recorded, actual = _read_population(directory)
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
        msg = f"{directory}: unreadable record or table: {type(exc).__name__}: {exc}"
        raise DatasetError(msg) from exc
    expected = _identity(run, jar_source)
    differing = sorted(key for key, value in expected.items() if record.get(key) != value)
    if differing:
        msg = f"{directory}: {RUN_RECORD_NAME} differs from this run in {', '.join(differing)}"
        raise DatasetError(msg)
    changed = sorted(name for name in TABLES if actual[name] != recorded[name])
    if changed:
        msg = f"{directory}: {', '.join(changed)} differ from the sha256 in {RUN_RECORD_NAME}"
        raise DatasetError(msg)


def _read_population(
    directory: Path,
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """Return the record, each table's recorded sha256, and each table's sha256 on disk.

    Raises:
        ValueError: If the record is not an object naming exactly `TABLES`.
    """
    record = json.loads((directory / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    tables = record["tables"]
    if not isinstance(tables, dict) or set(tables) != set(TABLES):
        msg = f"its tables are {tables!r}, not {list(TABLES)}"
        raise ValueError(msg)
    recorded = {name: tables[name]["sha256"] for name in TABLES}
    actual = {name: _sha256(directory / "csv" / name) for name in TABLES}
    return record, recorded, actual


def ensure_population(
    run: SyntheaRun,
    data_dir: Path,
    *,
    jar_source: Source = SYNTHEA_JAR,
    open_url: UrlOpener = open_https,
    resolve_java: Callable[[], tuple[Path, str]] = default_java,
    runner: Runner = run_java,
) -> Path:
    """Make the population `run` generates present under `data_dir`, and return its directory.

    A population that matches its record is reused with no Java and no download. One that does not
    is removed and generated again. Generation finds Java first, then fetches and verifies the jar,
    and runs it into a staging directory that is renamed into place once the CSVs exist.

    Args:
        run: The generation parameters.
        data_dir: The data root.
        jar_source: The pinned jar.
        open_url: The HTTPS opener for the jar.
        resolve_java: Returns the Java to use and its version line.
        runner: Runs Synthea's argv in a directory and returns its exit status.

    Returns:
        The population directory.

    Raises:
        UnavailableError: If Java, the jar or a finished run cannot be had.
        ContractError: If the jar differs from its pin, or Synthea's output lacks a table.
    """
    final = population_dir(run, data_dir, jar_source)
    log_copy = final.with_name(final.name + ".log")
    if final.is_dir():
        try:
            check_population(final, run, jar_source)
        except DatasetError as exc:
            logger.warning("regenerating population %s, which fails its record: %s", final, exc)
            shutil.rmtree(final)
        else:
            logger.info("population present path=%s", final)
            return final
    java, java_version = resolve_java()
    jar = ensure(jar_source, data_dir, open_url=open_url)
    logger.info("synthea java=%s version=%s", java, java_version)
    staging = staging_beside(final)
    try:
        record = _generate(run, staging, jar=jar, java=java, runner=runner, log_copy=log_copy)
        record |= _identity(run, jar_source) | {"java_version": java_version}
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        (staging / RUN_RECORD_NAME).write_text(text, encoding="utf-8")
        _publish(staging, final, run, jar_source)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log_copy.unlink(missing_ok=True)
    logger.info("generated population %s path=%s", final.name, final)
    return final


def _identity(run: SyntheaRun, jar_source: Source) -> dict[str, object]:
    """The record fields that must match for a population to be reused."""
    return {
        "config": list(CONFIG),
        "environment": dict(CHILD_ENVIRONMENT),
        "jar_sha256": jar_source.sha256,
        "jvm_options": list(JVM_OPTIONS),
        "run": asdict(run) | {"snapshot_date": run.snapshot_date.isoformat()},
    }


def _publish(staging: Path, final: Path, run: SyntheaRun, jar_source: Source) -> None:
    """Rename `staging` to `final`; if a concurrent run published first, check its population."""
    try:
        staging.rename(final)
    except OSError:
        if not final.is_dir():
            raise
        logger.info("population %s was published by a concurrent run; checking it", final)
        check_population(final, run, jar_source)


def _generate(
    run: SyntheaRun, staging: Path, *, jar: Path, java: Path, runner: Runner, log_copy: Path
) -> dict[str, object]:
    """Run Synthea into `staging` and return the tables part of the record.

    On any failure, Synthea's log is copied to `log_copy` and the error names it.
    """
    argv = synthea_argv(run, java=java, jar=jar, out_dir=staging)
    logger.info("synthea start log=%s argv=%s", staging / LOG_NAME, shlex.join(argv))
    started = datetime.now(UTC)
    try:
        tables = _run_synthea(argv, staging, runner)
    except DatasetError as exc:
        seconds = (datetime.now(UTC) - started).total_seconds()
        msg = f"{exc} after {seconds:.1f} s; {_keep_log(staging / LOG_NAME, log_copy)}"
        raise type(exc)(msg) from exc
    finished = datetime.now(UTC)
    logger.info(
        "synthea done population=%s seconds=%.1f",
        log_copy.stem,
        (finished - started).total_seconds(),
    )
    return {
        "finished_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "tables": tables,
    }


def _run_synthea(
    argv: Sequence[str], staging: Path, runner: Runner
) -> dict[str, dict[str, object]]:
    """Run `argv` and return each table's row count and sha256.

    Raises:
        UnavailableError: If Synthea times out or exits non-zero.
        ContractError: If it exits 0 without writing every table.
    """
    try:
        status = runner(argv, staging)
    except subprocess.TimeoutExpired as exc:
        msg = f"Synthea did not finish in {exc.timeout} s"
        raise UnavailableError(msg) from exc
    if status != 0:
        msg = f"Synthea exited {status}"
        raise UnavailableError(msg)
    tables: dict[str, dict[str, object]] = {}
    for name in TABLES:
        path = staging / "csv" / name
        if not path.is_file():
            msg = f"Synthea exited 0 but wrote no {name}"
            raise ContractError(msg)
        tables[name] = {"rows": _rows(path), "sha256": _sha256(path)}
    return tables


def _keep_log(log: Path, copy: Path) -> str:
    """Copy `log` to `copy` and say where it is, or why it could not be kept."""
    try:
        shutil.copyfile(log, copy)
    except OSError as exc:
        return f"its log could not be kept: {exc}"
    return f"its log is at {copy}"


def _rows(path: Path) -> int:
    """Return the data rows in the CSV at `path`, its header excluded."""
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
