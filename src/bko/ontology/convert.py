"""Convert a pinned OWL-only release to OBO with the pinned ROBOT jar, once, at fetch time.

`bko.ontology.releases.load_releases` sends a release whose `filename` ends `.owl` here and parses
the OBO it returns. The OBO sits beside the OWL in its snapshot directory, so a snapshot that
fails its pin is removed with it, and a later load reuses it. ROBOT's `convert` runs with no
reasoning and `--check false`, as the ELK comparison's does. The OWL is refused first if it names
an import, so ROBOT fetches nothing unpinned. Every load logs the OBO's size and sha256: the OWL
and the jar are pinned, the OBO is not (ADR `research/ontology-mapping-design`).

Java runs only through an `elk.Runner`, so a test passes a fake one.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol

from bko.datasets.errors import UnavailableError
from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.sources import ROBOT_JAR, Source
from bko.datasets.synthea_run import default_java
from bko.ontology import elk


logger = logging.getLogger(__name__)

CONVERT_MEMORY_GB: Final = 4
"""ROBOT's Java heap in GB for one conversion."""

CONVERT_TIMEOUT_SECONDS: Final = 600.0
"""How long one conversion may run before it is killed."""


class Convert(Protocol):
    """Make an OWL-only release's OBO present under the data root and return its path."""

    def __call__(self, source: Source, data_dir: Path, *, open_url: UrlOpener) -> Path:
        """Return the OBO converted from `source`, fetching and converting what is absent."""
        ...


def convert_argv(*, java: Path, jar: Path, owl: Path, out: Path) -> list[str]:
    """The command that converts `owl` to OBO at `out`, with no reasoning."""
    return [
        str(java),
        f"-Xmx{CONVERT_MEMORY_GB}g",
        "-jar",
        str(jar),
        "convert",
        "--input",
        str(owl),
        "--check",
        "false",
        "--format",
        "obo",
        "--output",
        str(out),
    ]


def converted(
    source: Source,
    data_dir: Path,
    *,
    open_url: UrlOpener,
    resolve_java: Callable[[], tuple[Path, str]] = default_java,
    robot: Source = ROBOT_JAR,
    runner: elk.Runner = elk.run_robot,
    timeout: float = CONVERT_TIMEOUT_SECONDS,
) -> Path:
    """Fetch `source`'s OWL file if absent, convert it to OBO beside it if absent, and return that.

    ROBOT writes into a hidden file beside the OBO, renamed into place once ROBOT exits 0, and its
    stderr is kept byte for byte at the OBO's path with `.stderr` added.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure`, for the OWL file or the jar.
        ReleaseFormatError: If the OWL file names an import; ROBOT does not run.
        UnavailableError: If there is no Java, or ROBOT times out, exits non-zero or writes no
            file; no OBO is left.
    """
    owl = ensure(source, data_dir, open_url=open_url)
    obo = owl.with_suffix(".obo")
    if obo.is_file():
        _log(source, obo, None)
        return obo
    elk.refuse_imports(owl)
    java, _ = resolve_java()
    jar = ensure(robot, data_dir, open_url=open_url)
    partial = obo.with_name(f".{obo.stem}.{secrets.token_hex(8)}.obo")
    kept = obo.with_name(obo.name + ".stderr")
    argv = convert_argv(java=java, jar=jar, owl=owl, out=partial)
    started = time.monotonic()
    try:
        try:
            status = runner(argv, timeout, owl.parent, kept)
        except subprocess.TimeoutExpired as exc:
            msg = f"ROBOT did not finish in {exc.timeout} s converting {owl}; {elk.tail(kept)}"
            raise UnavailableError(msg) from exc
        seconds = time.monotonic() - started
        if status != 0:
            msg = f"ROBOT exited {status} after {seconds:.1f} s converting {owl}; {elk.tail(kept)}"
            raise UnavailableError(msg)
        if not partial.is_file():
            msg = f"ROBOT exited 0 after {seconds:.1f} s but wrote no {partial}; {elk.tail(kept)}"
            raise UnavailableError(msg)
        partial.replace(obo)
    finally:
        partial.unlink(missing_ok=True)
    _log(source, obo, seconds)
    return obo


def _log(source: Source, obo: Path, seconds: float | None) -> None:
    """Log the OBO's identity: converted now, after `seconds`, or reused when `seconds` is None."""
    body = obo.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if seconds is None:
        logger.info(
            "converted release=%s path=%s bytes=%d sha256=%s reused=true",
            source.name,
            obo,
            len(body),
            digest,
        )
    else:
        logger.info(
            "converted release=%s seconds=%.1f path=%s bytes=%d sha256=%s reused=false",
            source.name,
            seconds,
            obo,
            len(body),
            digest,
        )
