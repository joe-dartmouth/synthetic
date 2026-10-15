"""Generate a Synthea population with the pinned jar, every output-changing parameter pinned.

ADR `research/synthea-ablation-design` records why each parameter is pinned. A population
directory `<data root>/synthea/<key>/` appears only through a rename after Synthea exits 0 and its
CSVs exist, so one that exists is complete. `run.json` in it records the parameters, the Java
version and each CSV's sha256 and row count, the identity `docs/EVIDENCE.md` asks an experiment to
cite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

from bko.datasets.errors import DatasetError, UnavailableError
from bko.datasets.fetch import UrlOpener, ensure, open_https, verify
from bko.datasets.sources import SYNTHEA_JAR, Source


logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: Final = 3600.0
"""How long one generation may run before it is killed."""

TABLES: Final = ("patients.csv", "conditions.csv")
"""The CSVs Synthea writes, and the only ones loaded."""

RECORD_NAME: Final = "run.json"
"""The file in a population directory that records how it was generated."""

LOG_NAME: Final = "synthea.log"
"""Synthea's combined output, kept in the population directory."""

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

type Runner = Callable[[Sequence[str], Path], int]
"""Run Synthea's argv in the given directory and return its exit status."""


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


DEFAULT_RUN: Final = SyntheaRun(
    seed=1, clinician_seed=1, population=2_000, snapshot_date=date(2026, 1, 1)
)
"""The population the command generates, sized from a pilot run so the demonstration ablation
cell holds at least 20 cases with probability about 0.95."""


def population_dir(run: SyntheaRun, data_dir: Path, jar_source: Source = SYNTHEA_JAR) -> Path:
    """Return the directory that holds the population `run` generates with `jar_source`."""
    key = (
        f"{jar_source.sha256[:12]}-s{run.seed}-cs{run.clinician_seed}-p{run.population}"
        f"-d{run.snapshot_date:%Y%m%d}-{run.state.lower()}"
    )
    return data_dir / "synthea" / key


def synthea_argv(run: SyntheaRun, *, java: Path, jar: Path, out_dir: Path) -> list[str]:
    """Return the command that generates `run` into `out_dir`."""
    stamp = f"{run.snapshot_date:%Y%m%d}"
    return [
        str(java),
        "-Duser.timezone=UTC",
        "-jar",
        str(jar),
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
    """Return `$JAVA_HOME/bin/java` when `JAVA_HOME` is set, else `java` on `PATH`.

    Raises:
        UnavailableError: If neither exists.
    """
    home = os.environ.get("JAVA_HOME")
    found = shutil.which("java")
    candidate = Path(home) / "bin" / "java" if home else (Path(found) if found else None)
    if candidate is None or not candidate.is_file():
        msg = "no java on JAVA_HOME or PATH; Synthea v4.0.0 needs JDK 17 or newer"
        raise UnavailableError(msg)
    return candidate


def probe_version(java: Path) -> str:
    """Return the first line `java -version` prints, which the JVM writes to stderr.

    Raises:
        UnavailableError: If it cannot run, times out, or exits non-zero, as the macOS stub does
            without a JDK.
    """
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [str(java), "-version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        msg = f"{java} -version failed: {exc}"
        raise UnavailableError(msg) from exc
    if result.returncode != 0:
        msg = f"{java} -version exited {result.returncode}: {result.stderr.strip()}"
        raise UnavailableError(msg)
    return result.stderr.splitlines()[0] if result.stderr else ""


def default_java() -> tuple[Path, str]:
    """Return the Java to run Synthea with and its version line."""
    java = find_java()
    return java, probe_version(java)


def run_java(argv: Sequence[str], cwd: Path, *, timeout: float = TIMEOUT_SECONDS) -> int:
    """Run `argv` in `cwd` with its output in `cwd/synthea.log`, and return its exit status.

    Raises:
        subprocess.TimeoutExpired: If it runs past `timeout`; it is killed first.
    """
    env = {name: os.environ[name] for name in ("PATH", "HOME", "LANG") if name in os.environ}
    env["TZ"] = "UTC"
    with (cwd / LOG_NAME).open("wb") as log:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return result.returncode


def check_population(directory: Path) -> None:
    """Check each CSV in `directory` against the sha256 its `run.json` records.

    Raises:
        DatasetError: If the record or a CSV is missing or a CSV differs from its record.
    """
    try:
        record = json.loads((directory / RECORD_NAME).read_text(encoding="utf-8"))
        recorded = {name: entry["sha256"] for name, entry in record["tables"].items()}
        actual = {name: _sha256(directory / "csv" / name) for name in recorded}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        msg = f"{directory}: unreadable record or table: {exc}"
        raise DatasetError(msg) from exc
    changed = sorted(name for name in recorded if actual[name] != recorded[name])
    if changed:
        msg = f"{directory}: {', '.join(changed)} differ from the sha256 in {RECORD_NAME}"
        raise DatasetError(msg)


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
    is removed and generated again. Generation fetches the jar if needed, verifies it, and runs it
    into a staging directory that is renamed into place once the CSVs exist.

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
        UnavailableError: If Java, the jar or a successful run cannot be had.
        ContractError: If the jar differs from its pin.
    """
    final = population_dir(run, data_dir, jar_source)
    if final.is_dir():
        try:
            check_population(final)
        except DatasetError as exc:
            logger.warning("regenerating a population that fails its record: %s", exc)
            shutil.rmtree(final)
        else:
            logger.info("population present path=%s", final)
            return final
    jar = ensure(jar_source, data_dir, open_url=open_url)
    java, java_version = resolve_java()
    verify(jar, jar_source)
    final.parent.mkdir(parents=True, exist_ok=True)
    # `mkdir` honours the umask, so the population gets the same mode as the directories around it.
    staging = final.parent / f".{final.name}.{secrets.token_hex(8)}"
    staging.mkdir()
    try:
        record = _generate(
            run,
            staging,
            jar=jar,
            java=java,
            runner=runner,
            log_copy=final.with_name(final.name + ".log"),
        )
        record |= {"jar_sha256": jar_source.sha256, "java_version": java_version}
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        (staging / RECORD_NAME).write_text(text, encoding="utf-8")
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info("generated path=%s", final)
    return final


def _generate(
    run: SyntheaRun, staging: Path, *, jar: Path, java: Path, runner: Runner, log_copy: Path
) -> dict[str, object]:
    argv = synthea_argv(run, java=java, jar=jar, out_dir=staging)
    logger.info("synthea start argv=%s", " ".join(argv))
    started = datetime.now(UTC)
    try:
        status = runner(argv, staging)
    except subprocess.TimeoutExpired as exc:
        msg = f"Synthea did not finish in {exc.timeout} s"
        raise UnavailableError(msg) from exc
    if status != 0:
        shutil.copyfile(staging / LOG_NAME, log_copy)
        msg = f"Synthea exited {status}; its log is at {log_copy}"
        raise UnavailableError(msg)
    tables: dict[str, dict[str, object]] = {}
    for name in TABLES:
        path = staging / "csv" / name
        if not path.is_file():
            msg = f"Synthea exited 0 but wrote no {name} under {staging}"
            raise UnavailableError(msg)
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle) - 1
        tables[name] = {"rows": rows, "sha256": _sha256(path)}
    finished = datetime.now(UTC)
    logger.info("synthea done seconds=%.1f", (finished - started).total_seconds())
    return {
        "config": list(CONFIG),
        "finished_at": finished.isoformat(),
        "run": asdict(run) | {"snapshot_date": run.snapshot_date.isoformat()},
        "started_at": started.isoformat(),
        "tables": tables,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
