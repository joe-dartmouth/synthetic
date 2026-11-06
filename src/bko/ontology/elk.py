"""Classify the pinned OWL releases with ELK through ROBOT and compare the result with the loader's.

`python -m bko.ontology elk` runs this. ROBOT reasons over each pinned OWL release with ELK, the
OWL 2 EL classifier, and writes the classified hierarchy as OBO; `compare_hierarchies` then sets
its `is_a` links beside those `bko.ontology.obo` loads from the same release's OBO file. Both
sides are compared as transitive ancestor sets, so a link one side states directly and the other
reaches through a chain is no difference. The scope is the mapped concepts and their ancestors
on either side; a link from a class outside it is ignored.

Java runs only through a `Runner`, so a test passes a fake one. The real runner gives Java the
environment `bko.datasets.synthea_run.child_environment` builds and keeps ROBOT's stderr byte for
byte beside the classified file.
"""

from __future__ import annotations

import logging
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from bko.datasets.errors import UnavailableError
from bko.datasets.sources import HPO_OWL, MONDO_OWL, Source
from bko.datasets.synthea_run import child_environment
from bko.ontology.errors import ReleaseFormatError
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)

ELK_MEMORY_GB: Final = 16
"""ROBOT's Java heap in GB: MONDO's classification peaked at 8.7 GB resident in the probe."""

ELK_TIMEOUT_SECONDS: Final = 7200.0
"""How long one classification may run before it is killed."""

OWL_RELEASES: Final[Mapping[str, Source]] = MappingProxyType({"HP": HPO_OWL, "MONDO": MONDO_OWL})
"""The OWL release ELK classifies for each concept prefix: the same release as the OBO pin."""

COLUMNS: Final = ("release", "child", "ancestor", "side")
"""The report's columns: a link `child` to `ancestor` that only `side` holds."""

IMPORTS: Final = (b"owl:imports", b"owl#imports")
"""How an OWL file names an import, in RDF/XML and in Turtle."""

_CHUNK: Final = 1 << 20

type Runner = Callable[[Sequence[str], float], tuple[int, bytes]]
"""Run ROBOT's argv with a timeout in seconds and return its exit status and stderr bytes."""

type Hierarchy = Mapping[str, Sequence[str]]
"""Each class's direct `is_a` targets."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """The links one hierarchy holds and the other lacks, within the scope.

    Attributes:
        concepts: The classes in scope: the given concepts and their ancestors on either side.
        only_loaded: `(child, ancestor)` links only the loaded release holds, sorted.
        only_elk: `(child, ancestor)` links only ELK's classification holds, sorted.
    """

    concepts: int
    only_loaded: tuple[tuple[str, str], ...]
    only_elk: tuple[tuple[str, str], ...]

    @property
    def differing(self) -> int:
        """How many links one side holds and the other lacks."""
        return len(self.only_loaded) + len(self.only_elk)


def compare_hierarchies(
    loaded: Hierarchy, classified: Hierarchy, concepts: Collection[str]
) -> Comparison:
    """Compare each in-scope class's transitive ancestors under `loaded` and `classified`.

    Args:
        loaded: The hierarchy the loader reads.
        classified: ELK's hierarchy.
        concepts: The mapped concepts; their ancestors on either side join the scope.

    Returns:
        The comparison; a cycle, as an equivalence makes, ends where it closes.
    """
    scope = set(concepts)
    for concept in concepts:
        scope |= _ancestors(loaded, concept) | _ancestors(classified, concept)
    only_loaded: list[tuple[str, str]] = []
    only_elk: list[tuple[str, str]] = []
    for child in sorted(scope):
        mine, theirs = _ancestors(loaded, child), _ancestors(classified, child)
        only_loaded += [(child, ancestor) for ancestor in sorted(mine - theirs)]
        only_elk += [(child, ancestor) for ancestor in sorted(theirs - mine)]
    return Comparison(len(scope), tuple(only_loaded), tuple(only_elk))


def loaded_hierarchy(ontology: Ontology) -> dict[str, tuple[str, ...]]:
    """Each term's direct `is_a` targets as the loader reads them."""
    return {curie: term.parents for curie, term in ontology.terms.items()}


def read_hierarchy(path: Path) -> dict[str, tuple[str, ...]]:
    """Each `[Term]` stanza's `is_a` targets in the OBO file at `path`, qualifiers dropped.

    Only `id:` and `is_a:` lines of term stanzas are read: ROBOT's output carries constructs, such
    as class expressions, that the comparison does not read.
    """
    found: dict[str, list[str]] = {}
    current: str | None = None
    in_term = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("["):
                in_term, current = line == "[Term]", None
            elif in_term and line.startswith("id:"):
                current = line.removeprefix("id:").split()[0]
                found.setdefault(current, [])
            elif in_term and current is not None and line.startswith("is_a:"):
                found[current].append(line.removeprefix("is_a:").split()[0])
    return {curie: tuple(sorted(set(parents))) for curie, parents in found.items()}


def robot_argv(*, java: Path, jar: Path, owl: Path, out: Path) -> list[str]:
    """The command that classifies `owl` with ELK and writes the result to `out` as OBO."""
    return [
        str(java),
        f"-Xmx{ELK_MEMORY_GB}g",
        "-jar",
        str(jar),
        "reason",
        "--reasoner",
        "ELK",
        "--equivalent-classes-allowed",
        "all",
        "--input",
        str(owl),
        "convert",
        "--check",
        "false",
        "--format",
        "obo",
        "--output",
        str(out),
    ]


def run_robot(argv: Sequence[str], timeout: float) -> tuple[int, bytes]:
    """Run `argv` with `child_environment` and no stdin; return its status and stderr bytes.

    Raises:
        subprocess.TimeoutExpired: If it runs past `timeout`; it is killed first.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        list(argv),
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stderr or b""


def refuse_imports(path: Path) -> None:
    """Refuse an OWL file that names an import, which ROBOT would fetch.

    Raises:
        ReleaseFormatError: If its bytes hold `owl:imports`.
    """
    overlap = max(len(marker) for marker in IMPORTS) - 1
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            window = tail + chunk
            if any(marker in window for marker in IMPORTS):
                msg = f"{path} names an owl:imports; a pinned release must be self-contained"
                raise ReleaseFormatError(msg)
            tail = window[-overlap:]


def classify(
    owl: Path,
    out: Path,
    *,
    java: Path,
    jar: Path,
    runner: Runner,
    timeout: float = ELK_TIMEOUT_SECONDS,
) -> float:
    """Classify `owl` into `out` and return the seconds it took.

    ROBOT's stderr is kept byte for byte at `out` with `.stderr` added.

    Raises:
        UnavailableError: If ROBOT times out, exits non-zero, or writes no `out`.
    """
    argv = robot_argv(java=java, jar=jar, owl=owl, out=out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    kept = out.with_name(out.name + ".stderr")
    started = time.monotonic()
    try:
        status, stderr = runner(argv, timeout)
    except subprocess.TimeoutExpired as exc:
        msg = f"ROBOT did not finish in {exc.timeout} s classifying {owl}"
        raise UnavailableError(msg) from exc
    seconds = time.monotonic() - started
    kept.write_bytes(stderr)
    if status != 0:
        msg = f"ROBOT exited {status} classifying {owl}; its stderr is at {kept}"
        raise UnavailableError(msg)
    if not out.is_file():
        msg = f"ROBOT exited 0 but wrote no {out}"
        raise UnavailableError(msg)
    return seconds


def rows(release: str, comparison: Comparison) -> Iterable[tuple[str, str, str, str]]:
    """The report's rows for one release: the loaded side's links, then ELK's."""
    for side, found in (("loaded", comparison.only_loaded), ("elk", comparison.only_elk)):
        for child, ancestor in found:
            yield release, child, ancestor, side


def _ancestors(hierarchy: Hierarchy, curie: str) -> set[str]:
    seen: set[str] = set()
    pending = [curie]
    while pending:
        for parent in hierarchy.get(pending.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                pending.append(parent)
    seen.discard(curie)
    return seen
