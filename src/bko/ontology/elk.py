"""Classify the pinned OWL releases with ELK through ROBOT and compare the result with the loader's.

`python -m bko.ontology elk` runs this. ROBOT reasons over each pinned OWL release with ELK, a
classifier for EL+bottom (EL with the bottom concept), part of the OWL 2 EL profile, and writes the
classified hierarchy as OBO; `compare_hierarchies` then sets its `is_a` links beside those
`bko.ontology.obo` loads from the same release's OBO file. Both sides are compared as transitive
ancestor sets, so a link one side states directly and the other reaches through a chain is no
difference, and each difference is an ancestor pair: a class and an ancestor only one side gives
it. The scope is the mapped concepts and their ancestors on either side; a pair whose class lies
outside it is ignored.

`read_hierarchy` is a reader of ROBOT's output, not of a release: pronto refuses the classified
HPO file, whose imported terms carry cross-references it cannot parse, so the `id:`, `is_a:` and
`equivalent_to:` lines of term stanzas are read by hand, an equivalence as `is_a` both ways (ADR
`research/ontology-mapping-design`, Ruled Out). The loaded OBO releases state no equivalence.

Java runs only through a `Runner`, so a test passes a fake one. The real runner gives Java the
environment `bko.datasets.synthea_run.child_environment` builds, runs it in the OWL file's
snapshot directory, and writes ROBOT's stderr byte for byte beside the classified file.
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
"""ROBOT's Java heap in GB, which the machine running `elk` must have free."""

ELK_TIMEOUT_SECONDS: Final = 7200.0
"""How long one classification may run before it is killed."""

OWL_RELEASES: Final[Mapping[str, Source]] = MappingProxyType({"HP": HPO_OWL, "MONDO": MONDO_OWL})
"""The OWL release ELK classifies for each concept prefix: the same release as the OBO pin."""

COLUMNS: Final = ("release", "child", "ancestor", "side")
"""The report's columns: an ancestor pair `child` to `ancestor` that only `side` holds."""

IMPORTS: Final = (b"owl:imports", b"owl#imports", b"Import(", b"<Import")
"""How an OWL file names an import: RDF/XML or Turtle, functional syntax, and OWL/XML."""

TAIL_LINES: Final = 5
"""How many of ROBOT's last stderr lines a failure's message quotes."""

_CHUNK: Final = 1 << 20

type Runner = Callable[[Sequence[str], float, Path, Path], int]
"""Run ROBOT's argv with a timeout in seconds in a working directory, writing its stderr to a
file, and return its exit status."""

type Hierarchy = Mapping[str, Sequence[str]]
"""Each class's direct `is_a` targets."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """The ancestor pairs one hierarchy holds and the other lacks, within the scope.

    Attributes:
        concepts: The classes in scope: the given concepts and their ancestors on either side.
        only_loaded: `(child, ancestor)` pairs only the loaded release holds, sorted.
        only_elk: `(child, ancestor)` pairs only ELK's classification holds, sorted.
        minimal_loaded: How many of `only_loaded` no other of them implies.
        minimal_elk: How many of `only_elk` no other of them implies.
        absent_loaded: How many classes in scope the loaded release has no term for.
        absent_elk: How many classes in scope ELK's classification has no term for.
    """

    concepts: int
    only_loaded: tuple[tuple[str, str], ...]
    only_elk: tuple[tuple[str, str], ...]
    minimal_loaded: int
    minimal_elk: int
    absent_loaded: int
    absent_elk: int

    @property
    def differing(self) -> int:
        """How many ancestor pairs one side holds and the other lacks."""
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
    return Comparison(
        concepts=len(scope),
        only_loaded=tuple(only_loaded),
        only_elk=tuple(only_elk),
        minimal_loaded=minimal(loaded, only_loaded),
        minimal_elk=minimal(classified, only_elk),
        absent_loaded=len(scope - loaded.keys()),
        absent_elk=len(scope - classified.keys()),
    )


def minimal(hierarchy: Hierarchy, pairs: Collection[tuple[str, str]]) -> int:
    """How many of `pairs` no other of them implies under `hierarchy`.

    `(c, a)` implies `(d, b)` when `d` is `c` or below it and `b` is `a` or above it, so one
    missing link counts once however many ancestor pairs it adds. Two pairs that imply each other,
    as an equivalence makes, both count.
    """
    up = {term: _ancestors(hierarchy, term) | {term} for pair in pairs for term in pair}

    def implies(first: tuple[str, str], second: tuple[str, str]) -> bool:
        return first[0] in up[second[0]] and second[1] in up[first[1]]

    return sum(
        not any(
            other != pair and implies(other, pair) and not implies(pair, other) for other in pairs
        )
        for pair in pairs
    )


def loaded_hierarchy(ontology: Ontology) -> dict[str, tuple[str, ...]]:
    """Each term's direct `is_a` targets as the loader reads them."""
    return {curie: term.parents for curie, term in ontology.terms.items()}


def read_hierarchy(path: Path) -> dict[str, tuple[str, ...]]:
    """Each `[Term]` stanza's `is_a` targets in ROBOT's OBO file at `path`, qualifiers dropped.

    Only `id:`, `is_a:` and `equivalent_to:` lines of term stanzas are read: ROBOT's output
    carries constructs, such as class expressions, that the comparison does not read. An
    equivalence is read as `is_a` both ways; an obsolete stanza is read as any other.

    Raises:
        ReleaseFormatError: If one of those lines has no value.
    """
    found: dict[str, set[str]] = {}
    current: str | None = None
    in_term = False
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line.startswith("["):
                in_term, current = line == "[Term]", None
                continue
            tag, _, value = line.partition(":")
            if not in_term or tag not in {"id", "is_a", "equivalent_to"}:
                continue
            words = value.split()
            if not words:
                msg = f"{path}:{number}: `{tag}:` has no value"
                raise ReleaseFormatError(msg)
            if tag == "id":
                current = words[0]
                found.setdefault(current, set())
            elif current is not None:
                found[current].add(words[0])
                if tag == "equivalent_to":
                    found.setdefault(words[0], set()).add(current)
    return {curie: tuple(sorted(parents)) for curie, parents in found.items()}


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


def run_robot(argv: Sequence[str], timeout: float, cwd: Path, stderr: Path) -> int:
    """Run `argv` in `cwd` with `child_environment` and no stdin, its stderr written to `stderr`.

    Returns:
        Its exit status.

    Raises:
        subprocess.TimeoutExpired: If it runs past `timeout`; it is killed first, and what it
            wrote to stderr stays in the file.
    """
    with stderr.open("wb") as kept:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            cwd=cwd,
            env=child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=kept,
            timeout=timeout,
            check=False,
        )
    return result.returncode


def refuse_imports(path: Path) -> None:
    """Refuse an OWL file whose bytes name an import, which ROBOT would fetch.

    A tripwire, not a parse: the sha256 pin is what guarantees the bytes, and this check catches
    a pinned release that names an import in any syntax `IMPORTS` lists, a literal holding one of
    those spellings included.

    Raises:
        ReleaseFormatError: If its bytes hold a spelling of `IMPORTS`.
    """
    overlap = max(len(marker) for marker in IMPORTS) - 1
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            window = tail + chunk
            if any(marker in window for marker in IMPORTS):
                msg = f"{path} names an import; a pinned release must be self-contained"
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

    ROBOT runs in `owl`'s directory, and its stderr is kept byte for byte at `out` with `.stderr`
    added; a stale `out` or stderr file is removed first.

    Raises:
        UnavailableError: If ROBOT times out, exits non-zero, or writes no `out`; the message
            names the stderr file and quotes its last `TAIL_LINES` lines.
    """
    argv = robot_argv(java=java, jar=jar, owl=owl, out=out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = out.with_name(out.name + ".stderr")
    out.unlink(missing_ok=True)
    kept.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        status = runner(argv, timeout, owl.parent, kept)
    except subprocess.TimeoutExpired as exc:
        msg = f"ROBOT did not finish in {exc.timeout} s classifying {owl}; {_tail(kept)}"
        raise UnavailableError(msg) from exc
    seconds = time.monotonic() - started
    if status != 0:
        msg = f"ROBOT exited {status} after {seconds:.1f} s classifying {owl}; {_tail(kept)}"
        raise UnavailableError(msg)
    if not out.is_file():
        msg = f"ROBOT exited 0 after {seconds:.1f} s but wrote no {out}; {_tail(kept)}"
        raise UnavailableError(msg)
    return seconds


def _tail(kept: Path) -> str:
    """Where ROBOT's stderr is, with its last `TAIL_LINES` lines."""
    if not kept.is_file():
        return f"no stderr at {kept}"
    lines = kept.read_bytes().decode("utf-8", errors="replace").splitlines()[-TAIL_LINES:]
    return f"its stderr is at {kept}, ending: {' | '.join(lines)}"


def rows(release: str, comparison: Comparison) -> Iterable[tuple[str, str, str, str]]:
    """The report's rows for one release: the loaded side's ancestor pairs, then ELK's."""
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
