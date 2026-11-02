"""Join holdout slices' scores into the scores one run writes: `python -m bko.evaluate merge`.

Each slice's scores file sits beside the `scores.log` its run wrote. Slices of one run hold the
same `data`, `versions` and `experiment` lines but for the keys a slice sets for itself
(`SLICE_KEYS`), so a slice at another training fraction, removal fraction, cutoff or commit is
refused.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from bko.evaluate.errors import EvaluationError, IdentityError, ScoresFormatError
from bko.evaluate.run_folder import ASSUMED, LINE, SCORES_LOG, keys
from bko.evaluate.scores import Measure, Scope, Score, Source, parse, render


logger = logging.getLogger(__name__)

type _Keys = frozenset[tuple[str, Source, Measure, Scope]]

SLICE_KEYS: Final = frozenset({"run", "holdouts", "seeds", "spend_ceiling"})
"""The `experiment` keys a slice sets for itself: its name, its holdouts and its budget."""

DATA_ASSUMED: Final = {"agreement": "none"}
"""The `data` keys an older line may lack, each with the value it is read as: open data."""

ADDED_DISTANCES: Final = frozenset({Measure.DCR_P5, Measure.DCR_MEDIAN})
"""The added rows' distance measures, which `score` writes for an arm only on a holdout where it
added rows (`bko.evaluate.run`), so one run's holdouts may differ in them."""

type _Lines = Mapping[str, Mapping[str, str] | str]
"""A slice log's `data` and `experiment` lines as keys and values, and its `versions` line."""


def merge(seeds: Sequence[int], paths: Sequence[Path]) -> str:
    """Join the scores files at `paths` into the bytes one run over `seeds` writes.

    Each file is read as `bko.evaluate.scores.parse` reads one, so a file with other columns is
    refused, and logged with its path, its lowest and highest holdout, its row count and its
    sha256. Every holdout must hold the same (arm, source, measure, scope) keys, as one run's
    holdouts do, but for the added rows' distances (`ADDED_DISTANCES`), which an arm that added
    no row on a holdout lacks there. Each file's sibling `scores.log` must hold the first file's
    `data`, `versions` and `experiment` lines, `SLICE_KEYS` set aside and a key an older line
    lacks read at the value `ASSUMED` or `DATA_ASSUMED` gives it, so slices scored under different
    code or configuration are refused. The rows are written in the order of `seeds` under one
    header, each holdout's rows in the order its file holds them, and the result is logged with
    its sha256.

    Raises:
        EvaluationError: If no file is given, two files hold the same holdout, a holdout holds
            other keys than the first file's first holdout, the files' holdouts are not
            exactly `seeds`, a file has no sibling `scores.log`, or its log holds no or several
            distinct `data`, `versions` or `experiment` lines, or other ones than the first's.
        ScoresFormatError: If a file is not UTF-8 or not the CSV `score` writes.
        OSError: If a file cannot be read.
    """
    if not paths:
        msg = "merge needs at least one scores file"
        raise EvaluationError(msg)
    owner: dict[int, Path] = {}
    scores: list[Score] = []
    reference: tuple[int, Path, _Keys] | None = None
    first: tuple[Path, _Lines] | None = None
    for path in paths:
        lines = _slice_lines(path.parent / SCORES_LOG)
        if first is None:
            first = (path.parent / SCORES_LOG, lines)
        else:
            _same(first, path.parent / SCORES_LOG, lines)
        data = path.read_bytes()
        try:
            read = parse(data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            msg = f"{path} is not UTF-8: {exc}"
            raise ScoresFormatError(msg) from exc
        held = sorted({s.holdout for s in read})
        logger.info(
            "merge file=%s holdouts=%s rows=%d sha256=%s",
            path,
            f"{held[0]}-{held[-1]}" if held else "none",
            len(read),
            hashlib.sha256(data).hexdigest(),
        )
        for holdout in held:
            if holdout in owner:
                msg = f"holdout {holdout} is in both {owner[holdout]} and {path}"
                raise EvaluationError(msg)
            owner[holdout] = path
            keys = frozenset(
                (s.arm, s.source, s.measure, s.scope)
                for s in read
                if s.holdout == holdout
                and not (s.source is Source.ADDED and s.measure in ADDED_DISTANCES)
            )
            if reference is None:
                reference = (holdout, path, keys)
            elif keys != reference[2]:
                msg = (
                    f"holdout {holdout} of {path} holds other scores than holdout {reference[0]} "
                    f"of {reference[1]}, so the files are not slices of one run"
                )
                raise EvaluationError(msg)
        scores.extend(read)
    missing = sorted(set(seeds) - owner.keys())
    extra = sorted(owner.keys() - set(seeds))
    if missing or extra:
        msg = f"the files' holdouts are not the run's: missing {missing}, not the run's {extra}"
        raise EvaluationError(msg)
    position = {seed: index for index, seed in enumerate(seeds)}
    ordered = sorted(scores, key=lambda s: position[s.holdout])
    text = render(ordered)
    logger.info(
        "merge files=%d holdouts=%d rows=%d sha256=%s",
        len(paths),
        len(owner),
        len(ordered),
        hashlib.sha256(text.encode()).hexdigest(),
    )
    return text


def _slice_lines(log: Path) -> _Lines:
    """A slice log's `data` and `experiment` lines' keys and values, and its `versions` line.

    Raises:
        EvaluationError: If `log` does not exist, holds a line that is not `key=value` where
            keys are read, or holds no or several distinct lines of one kind.
        OSError: If `log` cannot be read.
    """
    if not log.is_file():
        msg = f"{log} does not exist, so its slice's configuration cannot be checked"
        raise EvaluationError(msg)
    messages = [
        found["message"]
        for line in log.read_text(encoding="utf-8").splitlines()
        if (found := LINE.match(line)) is not None
    ]
    try:
        experiment = keys(_one(messages, "experiment run=", log), f"{log}'s")
    except IdentityError as exc:
        raise EvaluationError(str(exc)) from exc
    return {
        "data": {**DATA_ASSUMED, **_data_keys(_one(messages, "data dataset=", log))},
        "versions": _one(messages, "versions ", log),
        "experiment": {
            key: value for key, value in {**ASSUMED, **experiment}.items() if key not in SLICE_KEYS
        },
    }


def _one(messages: Sequence[str], start: str, log: Path) -> str:
    """The one distinct message of `messages` that begins with `start`.

    Raises:
        EvaluationError: If none does, or several distinct ones do.
    """
    distinct = sorted({m for m in messages if m.startswith(start)})
    if len(distinct) != 1:
        msg = f"{log} holds {len(distinct)} distinct {start.split(maxsplit=1)[0]} lines, not 1"
        raise EvaluationError(msg)
    return distinct[0]


def _data_keys(line: str) -> dict[str, str]:
    """A `data` line's keys and values; `agreement=` runs to the line's end, spaces and all."""
    head, _, agreement = line.partition(" agreement=")
    found = {key: value for key, _, value in (token.partition("=") for token in head.split()[1:])}
    if agreement:
        found["agreement"] = agreement
    return found


def _same(first: tuple[Path, _Lines], log: Path, lines: _Lines) -> None:
    """Refuse `log`'s lines unless they are the first log's.

    Raises:
        EvaluationError: Naming the first line kind and keys that differ.
    """
    reference, expected = first
    for kind, value in expected.items():
        other = lines[kind]
        if other == value:
            continue
        if isinstance(value, Mapping) and isinstance(other, Mapping):
            differ = sorted(k for k in value.keys() | other.keys() if value.get(k) != other.get(k))
            shown = ", ".join(f"{k}={value.get(k)} vs {other.get(k)}" for k in differ)
        else:
            shown = f"{value!r} vs {other!r}"
        msg = (
            f"{log}'s {kind} line differs from {reference}'s ({shown}), so the files are not "
            "slices of one run"
        )
        raise EvaluationError(msg)
