"""A run folder as `python -m bko.evaluate shared` reads it, and the checks across folders.

A run folder's `scores.log` names the data, the experiment, each holdout's split and each arm's
rows by sha256, and its `holdout-candidates.jsonl` holds every row a BKO arm kept. `read_folder`
reads both; `check_folders` holds identity checks 1, 2, 3 and 5 of ADR
`research/shared-test-split-reading` across the rescored folder and its `--with` folders.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.evaluate.data import Data
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import DIRECT_ARM, MESSAGES_ARMS
from bko.evaluate.holdout_candidates import Candidate, read
from bko.evaluate.utility import THRESHOLD


SCORES_LOG: Final = "scores.log"
CANDIDATES: Final = "holdout-candidates.jsonl"

BKO: Final = frozenset({*MESSAGES_ARMS, DIRECT_ARM})
"""The arms whose rows a candidates file holds."""

OWN_KEYS: Final = frozenset({"run", "arms", "replica", "spend_ceiling", "criteria"})
"""The `experiment` keys that name a run's own arms and budget, which two compared runs may differ
in; their baselines are held equal by their `arm` lines instead."""

ASSUMED: Final = MappingProxyType(
    {
        "frame": "uci",
        "removal": "none",
        "training_fraction": str(Fraction(1)),
        "decontaminate": "true",
        "threshold": str(UNDERREPRESENTED_BELOW),
        "cutoff": str(THRESHOLD),
        "replica_rows": "none",
    }
)
"""The `experiment` keys an older line may lack, each with the value it is read as.

Each names a setting added after a run of record was scored, at the value that run used: UCI's
frame, no removal, the whole training split, copies removed, E1's threshold, E1's cutoff and an
uncapped replica. `seeds` is also read, as `0-<holdouts - 1>`. Any other key a line lacks is
refused."""

LINE: Final = re.compile(r"^\S+ \S+ [A-Z]+ [\w.]+: (?P<message>.*)$")
ARM: Final = re.compile(
    r"arm holdout=(?P<holdout>\d+) arm=(?P<arm>\S+) attempt=(?P<attempt>\d+) seed=\d+ "
    r"added=(?P<added>\d+) sha256=(?P<sha256>\S+) params=(?P<params>.*)"
)
SPLIT: Final = re.compile(r"split seed=(?P<seed>\d+) ")
_DIGEST: Final = re.compile(r"data dataset=\S+ .*digest=(?P<digest>\w+)")
_SCORES: Final = re.compile(r"scores rows=\d+ sha256=(?P<sha256>\w+)")

type Key = tuple[int, str]
"""A holdout's seed and a name: an arm's, or a fill's."""


@dataclass(frozen=True, slots=True)
class Folder:
    """What a run folder's `scores.log` and candidates file say its scores were built from.

    Attributes:
        path: The folder.
        experiment: The `experiment` line's keys and values.
        data: The `data` line's digest, or the schema-2 candidates header's; None when neither
            names one, as for a schema-1 run.
        schema: The candidates file's schema; None with no candidates file.
        splits: Each holdout's `split` line, by seed.
        arms: Each `arm` line, by holdout and arm (a replica fill's arm ends `-replica`).
        lines: Each BKO arm's candidate lines, by holdout, arm and fill, in file order.
        log_sha256: The sha256 of `scores.log`.
        candidates_sha256: The sha256 of the candidates file; None when it is not read.
        scores_sha256: The `scores` line's sha256 of the run's `scores.csv`; None with no line.
    """

    path: Path
    experiment: Mapping[str, str]
    data: str | None
    schema: int | None
    splits: Mapping[int, str]
    arms: Mapping[Key, str]
    lines: Mapping[tuple[int, str, str], Sequence[Candidate]]
    log_sha256: str
    candidates_sha256: str | None
    scores_sha256: str | None

    def bko_fills(self) -> list[tuple[str, str, str]]:
        """Each BKO fill the `experiment` line asks for: its `arm` line's arm, its arm and fill."""
        gaps = [(arm, arm, "gap") for arm in self.listed("arms")]
        copy = [(f"{arm}-replica", arm, "replica") for arm in self.listed("replica")]
        return [fill for fill in (*gaps, *copy) if fill[1] in BKO]

    def listed(self, key: str) -> list[str]:
        """The arms the `experiment` line's comma-separated `key` names; none for an empty one."""
        return [arm for arm in self.experiment.get(key, "").split(",") if arm]

    def baselines(self) -> dict[Key, str]:
        """The `arm` lines of every arm but the BKO arms, their replica fills included."""
        return {key: line for key, line in self.arms.items() if base(key[1]) not in BKO}


def read_folder(path: Path) -> Folder:
    """Read a run folder's `scores.log` and, when it names a BKO arm, its candidates file.

    Raises:
        OSError: If a file is missing or cannot be read.
        IdentityError: If the log holds no `experiment` line or more than one, a token of it that
            is not `key=value`, two different lines for one split or arm, or two data digests.
        CandidatesFormatError: If the candidates file is not one.
    """
    log = (path / SCORES_LOG).read_bytes()
    messages = [
        found["message"]
        for line in log.decode("utf-8").splitlines()
        if (found := LINE.match(line)) is not None
    ]
    experiments = {m for m in messages if m.startswith("experiment run=")}
    if len(experiments) != 1:
        msg = f"{path / SCORES_LOG} holds {len(experiments)} distinct experiment lines, not 1"
        raise IdentityError(msg)
    splits: dict[int, str] = {}
    arms: dict[Key, str] = {}
    digests: set[str] = set()
    scores = None
    for message in messages:
        if (found := SPLIT.match(message)) is not None:
            _put(splits, int(found["seed"]), message, path)
        elif (found := ARM.fullmatch(message)) is not None:
            _put(arms, (int(found["holdout"]), found["arm"]), message, path)
        elif (found := _DIGEST.match(message)) is not None:
            digests.add(found["digest"])
        elif (found := _SCORES.match(message)) is not None:
            scores = found["sha256"]
    if len(digests) > 1:
        msg = f"{path / SCORES_LOG} names the data digests {sorted(digests)}"
        raise IdentityError(msg)
    data = next(iter(digests), None)
    schema, lines, candidates_sha256 = None, {}, None
    if any(base(arm) in BKO for _, arm in arms):
        data, schema, lines, candidates_sha256 = _candidates(path, data)
    experiment = keys(next(iter(experiments)), f"{path / SCORES_LOG}'s")
    log_sha256 = hashlib.sha256(log).hexdigest()
    return Folder(
        path, experiment, data, schema, splits, arms, lines, log_sha256, candidates_sha256, scores
    )


def _candidates(
    path: Path, data: str | None
) -> tuple[str | None, int, dict[tuple[int, str, str], list[Candidate]], str]:
    """The folder's candidates file: its data digest, schema, lines by fill and sha256.

    The digest is the header's for a schema-2 file, `data` (the log's) for a schema-1 file.

    Raises:
        OSError: If the file is missing or cannot be read.
        IdentityError: If the header names other data than the log's `data` line.
        CandidatesFormatError: If the file is not a candidates file.
    """
    raw = (path / CANDIDATES).read_bytes()
    candidates = read(raw)
    header = candidates.header
    schema = cast(int, header["schema"])  # `read` refuses a schema that is not an int
    if schema != 1:
        if data is not None and header["data"] != data:
            msg = f"{path}'s candidates header names data {header['data']}, its log {data}"
            raise IdentityError(msg)
        data = str(header["data"])
    grouped: defaultdict[tuple[int, str, str], list[Candidate]] = defaultdict(list)
    for line in candidates.candidates:
        grouped[line.holdout, line.arm, line.fill].append(line)
    return data, schema, dict(grouped), hashlib.sha256(raw).hexdigest()


def keys(line: str, whose: str) -> dict[str, str]:
    """An `experiment` line's keys and values.

    Raises:
        IdentityError: If a token after `experiment` is not `key=value`.
    """
    found: dict[str, str] = {}
    for token in line.split()[1:]:
        key, equals, value = token.partition("=")
        if not equals:
            msg = f"{whose} experiment line holds {token!r}, which is not key=value"
            raise IdentityError(msg)
        found[key] = value
    return found


def _put[K](found: dict[K, str], key: K, message: str, path: Path) -> None:
    if found.setdefault(key, message) != message:
        msg = f"{path / SCORES_LOG} logs two different lines for {key}: {found[key]!r}, {message!r}"
        raise IdentityError(msg)


def fields(line: str) -> re.Match[str]:
    """An `arm` line's fields.

    Raises:
        IdentityError: If `line` is not an `arm` line.
    """
    found = ARM.fullmatch(line)
    if found is None:
        msg = f"not an arm line: {line!r}"
        raise IdentityError(msg)
    return found


def base(arm: str) -> str:
    """An `arm` line's arm without a replica fill's `-replica` suffix."""
    return arm.removesuffix("-replica")


def check_folders(
    folder: Folder, others: Sequence[Folder], *, run: str, current: str, data: Data, uci: bool
) -> None:
    """Identity checks 1, 2, 3 and 5 across `folder` and `others`, before anything is scored.

    Each folder is read once, holds a `split` line per holdout its `experiment` line counts and,
    on each, an `arm` line per arm and replica fill that line names, so a line format that drifts
    from `ARM` or `SPLIT` is refused rather than skipped.

    Args:
        folder: The rescored run's folder.
        others: The `--with` folders.
        run: The named run rescored; the folder's `experiment` line names it.
        current: The `experiment` line the named run logs now.
        data: The loaded data; every folder's digest is its digest.
        uci: Whether the named run scores UCI Heart Disease, which a schema-1 candidates file
            holds (`bko.evaluate.holdout_candidates.read` checks its archive digest).

    Check 2 holds every `experiment` key equal but `models`, whose current list may extend the
    folder's (`_extends`), so `bko.evaluate.panel` reads a folder scored at fewer models.

    Raises:
        IdentityError: If a check fails.
    """
    seen: set[Path] = set()
    for each in (folder, *others):
        resolved = each.path.resolve()
        if resolved in seen:
            msg = f"{each.path} is read twice: a --with folder repeats --folder or another --with"
            raise IdentityError(msg)
        seen.add(resolved)
        found = each.data
        if found is None and each.schema == 1 and uci:
            found = data.digest
        if found != data.digest:
            msg = f"{each.path} holds the data digest {found}, not the loaded data's {data.digest}"
            raise IdentityError(msg)
    if folder.experiment.get("run") != run:
        msg = f"{folder.path} is run {folder.experiment.get('run')!r}'s, not {run!r}'s"
        raise IdentityError(msg)
    now = keys(current, f"the run {run!r}'s")
    _same_keys(folder.experiment, now, {"models"}, f"{folder.path} and the run {run!r} now")
    theirs, ours = folder.experiment.get("models", ""), now.get("models", "")
    if not _extends(theirs, ours):
        msg = f"{folder.path}'s models={theirs} is not a prefix of the run {run!r}'s models={ours}"
        raise IdentityError(msg)
    for each in (folder, *others):
        _complete(each)
    for other in others:
        _same_keys(folder.experiment, other.experiment, OWN_KEYS, f"{folder.path} and {other.path}")
        if other.splits != folder.splits:
            msg = f"{other.path}'s split lines differ from {folder.path}'s"
            raise IdentityError(msg)
        if other.baselines() != folder.baselines():
            differing = sorted(set(other.baselines().items()) ^ set(folder.baselines().items()))
            msg = f"{other.path}'s baseline arm lines differ from {folder.path}'s: {differing[:2]}"
            raise IdentityError(msg)


def _extends(folder_models: str, current_models: str) -> bool:
    """Whether the comma-separated `current_models` list starts with all of `folder_models`.

    An empty folder list extends to nothing, so a folder that names no models is refused.
    """
    theirs, ours = folder_models.split(","), current_models.split(",")
    return bool(folder_models) and ours[: len(theirs)] == theirs


def _complete(folder: Folder) -> None:
    """Refuse a folder missing a holdout's `split` line, or an `arm` line its experiment names.

    Raises:
        IdentityError: If the folder logs `split` lines for another number of holdouts than its
            `experiment` line's `holdouts`, or a holdout's `arm` lines name other arms than its
            `arms`, and its `replica` arms with `-replica`.
    """
    where = folder.path / SCORES_LOG
    holdouts = folder.experiment.get("holdouts")
    if str(len(folder.splits)) != holdouts:
        msg = f"{where} logs split lines for {len(folder.splits)} holdouts, not its experiment"
        msg += f" line's {holdouts}"
        raise IdentityError(msg)
    expected = {*folder.listed("arms"), *(f"{arm}-replica" for arm in folder.listed("replica"))}
    for holdout in sorted(folder.splits):
        found = {arm for seed, arm in folder.arms if seed == holdout}
        if found != expected:
            msg = f"{where} logs the arms {sorted(found)} on holdout {holdout}, not its experiment"
            msg += f" line's {sorted(expected)}"
            raise IdentityError(msg)


def _same_keys(
    ours: Mapping[str, str], theirs: Mapping[str, str], skip: set[str] | frozenset[str], where: str
) -> None:
    """Refuse two `experiment` lines that differ on a key outside `skip`.

    A key one line lacks is read at its `ASSUMED` value, so the other line must hold that value.

    Raises:
        IdentityError: If a value differs, or one line lacks a key with no assumed value.
    """
    differing: dict[str, tuple[str, str]] = {}
    for key in sorted((set(ours) | set(theirs)) - skip):
        pair = (_value(ours, key, where), _value(theirs, key, where))
        if pair[0] != pair[1]:
            differing[key] = pair
    if differing:
        msg = f"the experiment lines of {where} differ: {differing}"
        raise IdentityError(msg)


def _value(line: Mapping[str, str], key: str, where: str) -> str:
    """`line`'s value of `key`, or the value a line that lacks it is read as."""
    if key in line:
        return line[key]
    if key in ASSUMED:
        return ASSUMED[key]
    if key == "seeds" and line.get("holdouts", "").isdigit():
        return f"0-{int(line['holdouts']) - 1}"
    msg = f"the experiment lines of {where}: only one names {key}, which has no assumed value"
    raise IdentityError(msg)
