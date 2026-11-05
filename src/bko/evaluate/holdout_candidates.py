"""The per-holdout candidates file: every candidate a run's BKO arms generate, with its outcome.

The file is JSONL. Its first line is the header, `kind` `header`: the schema, the named run, the
name of its `frame`, the sha256 of its records under `data` (`bko.evaluate.data.Data.digest`), the
sha256 of each ontology release, and the generator settings, which a run with no Messages arm
omits. A schema-1 file, written before frames, names the UCI snapshot's sha256 under `uci` in place
of `frame` and `data`, and `read` checks it against the pinned archive. Every other line
is one candidate, `kind` `candidate`: the holdout, the arm, the fill (`gap` or `replica`), the
attempt within that fill counted from 1 and whether it `returned` or was refused, the call (null for
a direct arm's record, drawn with no call) and the cell it asked for, the candidate's 0-based
position among the call's, then either the decoded `candidate` or the reason it is `undecoded`, its
`outcome` and the checks it `failed`. The outcome
is `kept`, `rejected` (by `bko.validate`, with its failed checks), or `dropped-range`,
`dropped-cell`, `dropped-concepts` (a prescribing arm's alone), `dropped-curation` (a curating
arm's alone) or `dropped-quota` (by `bko.baselines.records.drops`), or `dropped-likelihood` (a
ranking arm's alone, by `bko.evaluate.attempts.Attempt.rows`). A schema-3 line of a scoring arm
also holds the `score` its curator read: `confidence` and `aleatoric` on a curating arm's line that
passed validation, `likelihood` on a ranking arm's line it ranked, null for minus infinity. Each
(arm, holdout, fill)'s lines count what its candidate scores count.

An arm writes a fill's lines when the harness drains it, so a stopped run's file holds the fills
drained before the stop. Every line's keys are sorted and no line holds a time, so a replay of a
run's call log rewrites its file byte for byte. ADR `research/e1-evaluation-protocol` records the
file; `docs/DEVELOPMENT.md` (Evaluation) holds the commands that write it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Final, NoReturn, Self, TextIO

from bko.baselines.records import Drop
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.sources import Source as Download
from bko.evaluate.data import Data
from bko.evaluate.errors import CandidatesFormatError
from bko.evaluate.scores import Source
from bko.generate.messages import Settings


SCHEMA: Final = 3
"""The file format's version, written in the header."""

FILLS: Final[Mapping[Source, str]] = MappingProxyType(
    {Source.CANDIDATES: "gap", Source.REPLICA_CANDIDATES: "replica"}
)
"""The fill each candidate-count source counts, as a line names it."""

KEPT: Final = "kept"
"""The outcome of a candidate that passed every check and counts toward its cell."""

REJECTED: Final = "rejected"
"""The outcome of a candidate that failed a check, an undecoded record included."""

OUTCOMES: Final = (KEPT, REJECTED, *(f"dropped-{drop}" for drop in Drop))
"""Every outcome a line names."""

_HEADERS: Final[Mapping[int, frozenset[str]]] = MappingProxyType(
    {
        1: frozenset({"kind", "schema", "run", "uci", "releases"}),
        2: frozenset({"kind", "schema", "run", "frame", "data", "releases"}),
        SCHEMA: frozenset({"kind", "schema", "run", "frame", "data", "releases"}),
    }
)
"""Each readable schema's header keys besides `settings`: schema 1, the judged run's, names UCI's
archive digest under `uci`; schemas 2 and 3 name the frame and the records' digest, and only a
schema-3 line may hold a `score`."""

_SETTINGS: Final = "settings"

_SCORE: Final = "score"

_LINE: Final = frozenset(
    {
        "kind",
        "holdout",
        "arm",
        "fill",
        "attempt",
        "returned",
        "call",
        "cell",
        "position",
        "outcome",
        "failed",
    }
)


@dataclass(frozen=True, slots=True)
class Origin:
    """What a candidates file's header names besides the settings.

    Attributes:
        run: The named run, or `app` for an app fill.
        data: The data the run scored; the header names its frame and digest.
        releases: The ontology releases its gap detection read, by concept prefix.
    """

    run: str
    data: Data
    releases: Mapping[str, Download]


@dataclass(frozen=True, slots=True)
class Seen:
    """One candidate as an attempt took it from a call.

    Attributes:
        call: The call's index in the run, counted from 1; None for a record drawn with no call.
        cell: The cell the call asked for, its parts joined by `/`.
        position: Its 0-based position among the call's candidates.
        candidate: The decoded item; None for an undecoded record.
        undecoded: Why the record has no decoded item, or None.
        outcome: One of `OUTCOMES`.
        failed: The checks it failed, for a rejected candidate.
        score: What its arm's curator read of it, by name, None for minus infinity; None for a
            candidate no curator read.
    """

    call: int | None
    cell: str
    position: int
    candidate: object
    undecoded: str | None
    outcome: str
    failed: tuple[str, ...]
    score: Mapping[str, float | None] | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate line.

    Attributes:
        holdout: The holdout's seed.
        arm: The BKO arm.
        fill: `gap` or `replica`.
        attempt: The attempt within the fill, counted from 1.
        returned: Whether the attempt returned a fill rather than being refused.
        seen: The candidate as its attempt took it.
    """

    holdout: int
    arm: str
    fill: str
    attempt: int
    returned: bool
    seen: Seen

    @property
    def call(self) -> int | None:
        """The call's index in the run, or None for a record drawn with no call."""
        return self.seen.call

    @property
    def cell(self) -> str:
        """The cell the call asked for."""
        return self.seen.cell

    @property
    def position(self) -> int:
        """The candidate's position among the call's."""
        return self.seen.position

    @property
    def candidate(self) -> object:
        """The decoded item, or None for an undecoded record."""
        return self.seen.candidate

    @property
    def undecoded(self) -> str | None:
        """Why the record did not decode, or None."""
        return self.seen.undecoded

    @property
    def outcome(self) -> str:
        """One of `OUTCOMES`."""
        return self.seen.outcome

    @property
    def failed(self) -> tuple[str, ...]:
        """The checks a rejected candidate failed."""
        return self.seen.failed

    @property
    def score(self) -> Mapping[str, float | None] | None:
        """What its arm's curator read of it, or None."""
        return self.seen.score

    def line(self) -> dict[str, object]:
        """The candidate as its line's JSON object."""
        seen = self.seen
        item = (
            {"candidate": seen.candidate}
            if seen.undecoded is None
            else {"undecoded": seen.undecoded}
        )
        return {
            "kind": "candidate",
            "holdout": self.holdout,
            "arm": self.arm,
            "fill": self.fill,
            "attempt": self.attempt,
            "returned": self.returned,
            "call": seen.call,
            "cell": seen.cell,
            "position": seen.position,
            **item,
            "outcome": seen.outcome,
            "failed": list(seen.failed),
            **({} if seen.score is None else {_SCORE: dict(seen.score)}),
        }


@dataclass(frozen=True, slots=True)
class HoldoutCandidates:
    """A candidates file, read.

    Attributes:
        header: Its header line.
        candidates: Its candidate lines, in file order.
    """

    header: Mapping[str, object]
    candidates: tuple[Candidate, ...]


def header(origin: Origin, settings: Settings | None) -> dict[str, object]:
    """The header line a run with `origin` and `settings` writes; no settings key for None."""
    line: dict[str, object] = {
        "kind": "header",
        "schema": SCHEMA,
        "run": origin.run,
        "frame": origin.data.frame.name,
        "data": origin.data.digest,
        "releases": {prefix: source.sha256 for prefix, source in origin.releases.items()},
    }
    if settings is not None:
        line[_SETTINGS] = dataclasses.asdict(settings)
    return line


class CandidateFile:
    """One run's candidates file, created exclusively and readable by its owner only.

    Attributes:
        path: The file.
    """

    def __init__(self, path: Path, origin: Origin, settings: Settings | None) -> None:
        """Create the file and write its header; `settings` is None for a run with no Messages arm.

        Raises:
            FileExistsError: If `path` exists.
            OSError: If the file cannot be opened or its header written; the file is then removed.
        """
        self.path = path
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        opened: TextIO | None = None
        try:
            opened = os.fdopen(descriptor, "w", encoding="utf-8")
            self._file: TextIO = opened
            self._write([header(origin, settings)])
        except BaseException:
            if opened is None:
                os.close(descriptor)
            else:
                opened.close()
            path.unlink(missing_ok=True)
            raise

    def __enter__(self) -> Self:
        """Return the file."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file; never suppress."""
        self._file.close()

    def write(self, candidates: Iterable[Candidate]) -> None:
        """Append one line per candidate, and flush.

        Raises:
            ValueError: If a candidate holds a number that is not finite; nothing is written.
        """
        self._write([candidate.line() for candidate in candidates])

    def discard(self) -> None:
        """Close and remove the file, so a run that never started leaves none to block a rerun."""
        self._file.close()
        self.path.unlink(missing_ok=True)

    def _write(self, lines: list[dict[str, object]]) -> None:
        text = "".join(json.dumps(line, sort_keys=True, allow_nan=False) + "\n" for line in lines)
        self._file.write(text)
        self._file.flush()


def read(data: bytes) -> HoldoutCandidates:
    """Read a candidates file's bytes.

    Raises:
        CandidatesFormatError: If the bytes are not a header line then candidate lines, each as
            this module writes it.
    """
    try:
        lines = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, ValueError) as exc:
        msg = f"a candidates file is UTF-8 JSON lines: {exc}"
        raise CandidatesFormatError(msg) from None
    if not lines:
        msg = "a candidates file starts with its header line; this one is empty"
        raise CandidatesFormatError(msg)
    first, *rest = lines
    if not isinstance(first, dict) or first.get("kind") != "header":
        msg = "a candidates file's first line is its header"
        raise CandidatesFormatError(msg)
    # The schema first, so an older file says its schema rather than its keys.
    schema = first.get("schema")
    if type(schema) is not int or schema not in _HEADERS:
        msg = f"the candidates file's schema is {schema!r}, not one of {sorted(_HEADERS)}"
        raise CandidatesFormatError(msg)
    keys = _HEADERS[schema]
    if set(first) - {_SETTINGS} != keys:
        msg = (
            f"a schema-{schema} candidates file's header has the keys {sorted(keys)} "
            f"and, for a run with a Messages arm, {_SETTINGS!r}"
        )
        raise CandidatesFormatError(msg)
    if schema == 1 and first["uci"] != UCI_HEART_DISEASE.sha256:
        msg = f"the schema-1 header's uci digest is not the pinned archive's: {first['uci']!r}"
        raise CandidatesFormatError(msg)
    scored = schema == SCHEMA
    return HoldoutCandidates(
        MappingProxyType(first),
        tuple(_candidate(number, line, scored=scored) for number, line in enumerate(rest, start=2)),
    )


def _candidate(number: int, line: object, *, scored: bool) -> Candidate:
    if not isinstance(line, dict) or line.get("kind") != "candidate":
        _refuse(number, "is not a candidate line")
    item = set(line) - _LINE - ({_SCORE} if scored else set())
    if not set(line) >= _LINE or item not in ({"candidate"}, {"undecoded"}):
        _refuse(number, f"holds the keys {sorted(line)}")
    undecoded = line.get("undecoded")
    failed = line["failed"]
    checks = (
        _count(line["holdout"], 0),
        isinstance(line["arm"], str),
        line["fill"] in FILLS.values(),
        _count(line["attempt"], 1),
        isinstance(line["returned"], bool),
        line["call"] is None or _count(line["call"], 1),
        isinstance(line["cell"], str),
        _count(line["position"], 0),
        "undecoded" not in line or isinstance(undecoded, str),
        line["outcome"] in OUTCOMES,
        isinstance(failed, list) and all(isinstance(check, str) for check in failed),
        _SCORE not in line or _score(line[_SCORE]),
    )
    if not all(checks):
        _refuse(number, "holds a value of the wrong type")
    if bool(failed) != (line["outcome"] == REJECTED):
        _refuse(number, "names failed checks unless it is rejected, or none when it is")
    if undecoded is not None and line["outcome"] != REJECTED:
        _refuse(number, "is undecoded, so its outcome is rejected")
    seen = Seen(
        line["call"],
        line["cell"],
        line["position"],
        line.get("candidate"),
        undecoded,
        line["outcome"],
        tuple(failed),
        line.get(_SCORE),
    )
    return Candidate(
        line["holdout"], line["arm"], line["fill"], line["attempt"], line["returned"], seen
    )


def _score(value: object) -> bool:
    """Whether `value` is a score: names, each with a finite number or null."""
    return isinstance(value, dict) and all(
        number is None or _finite(number) for number in value.values()
    )


def _finite(number: object) -> bool:
    """Whether `number` is an int or float, not a bool, that reads as a finite float."""
    if not isinstance(number, int | float) or isinstance(number, bool):
        return False
    try:
        return math.isfinite(number)
    except OverflowError:
        # An int past a float's range, which no score reaches.
        return False


def _count(value: object, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _refuse(number: int, reason: str) -> NoReturn:
    msg = f"line {number} of the candidates file {reason}"
    raise CandidatesFormatError(msg)
