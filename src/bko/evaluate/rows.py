"""`python -m bko.evaluate rows`: rescore a run's folder and store every fill it checks.

`store` rescores a recorded folder through `bko.evaluate.shared.rescore`, at the folder's own
models, and makes no API call. After a holdout passes identity checks 3 and 4, `compare_holdout`
holds it to the strict check 7 against the folder's `scores.csv`; only then does `RowsFile.write`
append every fill of that holdout to the store, so a holdout that fails a check (exit 3) stores
nothing. ADR `research/rebuilt-rows-store` holds the store's format and what a reader checks;
ADR `research/shared-test-split-reading` holds the strict check 7 and its 50,000-row exception.

`bko.evaluate.rows_read` reads a store back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Self, TextIO

import numpy as np

from bko.baselines.methods import Runner
from bko.evaluate.classifier import Factory, Model
from bko.evaluate.data import Data, digested
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import Experiment
from bko.evaluate.filling import Filled, Logged
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.rows_read import SCHEMA
from bko.evaluate.run_folder import Folder, fields
from bko.evaluate.scores import Measure, Score, Source, parse, render
from bko.evaluate.shared import rescore


logger = logging.getLogger("bko.evaluate")

SCORES_CSV: Final = "scores.csv"

SUBSAMPLED_ABOVE: Final = 50_000
"""The training rows above which a folder's replica fidelity may hold one subsample draw."""

SUBSAMPLED_MEASURES: Final = frozenset({Measure.QUALITY, Measure.COLUMN_PAIR_TRENDS})
"""The replica measures a subsampled holdout's rescore may read differently."""

_TRAIN: Final = re.compile(r" train=(?P<train>\d+) ")

type ScoreKey = tuple[int, str, str, str, str]
"""A score's holdout, arm, source, measure and scope, as the scores CSV writes them."""


def creatable(path: Path) -> None:
    """Refuse a store path that exists or whose directory does not, before any work.

    Raises:
        FileExistsError: If `path` exists.
        FileNotFoundError: If its directory does not exist.
    """
    if path.exists():
        msg = f"the store {path} exists"
        raise FileExistsError(msg)
    if not path.parent.is_dir():
        msg = f"the store's directory {path.parent} does not exist"
        raise FileNotFoundError(msg)


class RowsFile:
    """A store being written, created on the first holdout it stores, mode 0600 and `O_EXCL`.

    Creating it lazily keeps a folder refused before any holdout passes from leaving a store.
    """

    def __init__(self, path: Path, header: Mapping[str, object]) -> None:
        """Hold `header` for the store at `path`, which `creatable` checks now.

        Raises:
            FileExistsError, FileNotFoundError: As `creatable`.
        """
        creatable(path)
        self.path, self._header = path, header
        self._file: TextIO | None = None
        self._written: set[tuple[int, str]] = set()
        self.holdouts = 0

    @property
    def stored(self) -> int:
        """The fill lines stored."""
        return len(self._written)

    def __enter__(self) -> Self:
        """The store."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the store; on an error, log the `rows partial` WARNING if it was created."""
        if self._file is None:
            return
        self._file.close()
        if error is not None:
            logger.warning("rows partial path=%s holdouts=%d", self.path, self.holdouts)

    def open(self) -> TextIO:
        """The store's file, created with its header, synced with its directory, on first use.

        Raises:
            FileExistsError: If the store's path exists by now.
            OSError: If it cannot be created or written; the file is closed first.
        """
        if self._file is not None:
            return self._file
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            file = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            file.write(_json(self._header))
            _sync(file)
            _sync_directory(self.path.parent)
        except BaseException:
            file.close()
            raise
        self._file = file
        return file

    def write(
        self, holdout: int, fills: Sequence[Filled], *, arms: Mapping[tuple[int, str], str]
    ) -> int:
        """Append every fill of `holdout` in one write, then flush and sync; return how many.

        Every line is built and checked before any is written, and a fill counts as stored only
        once the sync returns. Each fill's `rows ... matches_arm=true` line is logged after it.

        Args:
            holdout: The holdout's seed.
            fills: Its fills.
            arms: The folder's `arm` lines by holdout and arm; each fill's line must be its own.

        Raises:
            IdentityError: If a fill holds a non-finite value, its rows' sha256 or count is not its
                `arm` line's, its `arm` line is not the folder's, or its holdout and arm are
                already stored.
            OSError: As `open`, or if the write or sync fails.
        """
        lines = [_line(holdout, fill) for fill in fills]
        keys = [(holdout, fill.name) for fill in fills]
        for key, fill in zip(keys, fills, strict=True):
            if arms.get(key) != fill.line:
                msg = f"holdout {holdout}'s {fill.name} line {fill.line!r} is not the folder's"
                msg += f" arm line {arms.get(key)!r}"
                raise IdentityError(msg)
        if len(set(keys)) != len(keys) or self._written & set(keys):
            msg = f"holdout {holdout} names a fill twice, or one already stored: {keys}"
            raise IdentityError(msg)
        file = self.open()
        file.write("".join(_json(line) for line in lines))
        _sync(file)
        self._written |= set(keys)
        self.holdouts += 1
        for line in lines:
            logger.info(
                "rows holdout=%d arm=%s added=%d sha256=%s matches_arm=true",
                holdout,
                line["arm"],
                line["added"],
                line["sha256"],
            )
        return len(lines)


def _json(line: Mapping[str, object]) -> str:
    """`line` as one store line: sorted keys, no spaces, no NaN."""
    return json.dumps(line, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _sync(file: TextIO) -> None:
    """Flush `file` and sync it to disk, so a power loss keeps each written holdout."""
    file.flush()
    os.fsync(file.fileno())


def _sync_directory(directory: Path) -> None:
    """Sync `directory`, so a power loss keeps the store's new entry in it."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _line(holdout: int, fill: Filled) -> dict[str, object]:
    """`fill`'s store line, checked against its `arm` line."""
    added = fill.baseline.added
    numbers = added.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(numbers).all():
        msg = f"holdout {holdout}'s {fill.name} rows hold a non-finite value"
        raise IdentityError(msg)
    csv, sha = digested(added)
    logged = fields(fill.line)
    if (str(holdout), fill.name, str(len(added)), sha) != logged.group(
        "holdout", "arm", "added", "sha256"
    ):
        msg = f"holdout {holdout}'s {fill.name} rows ({len(added)}, {sha}) are not its arm line"
        msg += f" {fill.line!r}"
        raise IdentityError(msg)
    return {
        "kind": "rows",
        "holdout": holdout,
        "arm": fill.name,
        "attempt": fill.attempt,
        "added": len(added),
        "sha256": sha,
        "csv": csv,
    }


def subsampled(split: str | None) -> bool:
    """Whether the holdout whose `split` line this is trains on more than `SUBSAMPLED_ABOVE` rows.

    Raises:
        IdentityError: If there is no split line, or it names no training rows.
    """
    found = None if split is None else _TRAIN.search(split)
    if found is None:
        msg = f"the split line {split!r} names no training rows"
        raise IdentityError(msg)
    return int(found["train"]) > SUBSAMPLED_ABOVE


def held_out(key: ScoreKey) -> bool:
    """Whether `rows` admits `key` beyond the folder's: a held-out row's key."""
    return key[2] == Source.HELD_OUT.value


def compare_holdout(
    ours: Sequence[Score],
    theirs: Sequence[Score],
    *,
    holdout: int,
    admitted: Callable[[ScoreKey], bool],
    sampled: bool,
) -> int:
    """The strict check 7 on one holdout; return how many replica fidelity rows differ.

    Each key's rendered line is compared, so NaN matches NaN. A key only `ours` holds must be
    `admitted`; a replica's `quality` and `column-pair-trends` may differ when `sampled`, the
    holdout's fidelity read from a subsample (`subsampled`), and each such row is logged as
    `rows differing holdout=%d key=%s ours=%s theirs=%s`, its key as the scores CSV writes it.

    Raises:
        IdentityError: On any other difference, or a key only `theirs` holds.
    """
    mine, folder = _rendered(ours, holdout), _rendered(theirs, holdout)
    differing = 0
    for key in sorted(set(mine) | set(folder)):
        if key not in folder:
            if admitted(key):
                continue
            msg = f"holdout {holdout}'s score {key}: the folder holds no score for it"
            raise IdentityError(msg)
        if key not in mine:
            msg = f"holdout {holdout}'s score {key}: the rescore holds no score for it"
            raise IdentityError(msg)
        if mine[key] == folder[key]:
            continue
        loose = key[2] == Source.REPLICA.value and Measure(key[3]) in SUBSAMPLED_MEASURES
        if sampled and loose:
            differing += 1
            logger.info(
                "rows differing holdout=%d key=%s ours=%s theirs=%s",
                holdout,
                ",".join(map(str, key)),
                mine[key],
                folder[key],
            )
            continue
        msg = f"holdout {holdout}'s score {key} reads {mine[key]!r}, the folder's"
        msg += f" {folder[key]!r}"
        raise IdentityError(msg)
    return differing


def _rendered(scores: Sequence[Score], holdout: int) -> dict[ScoreKey, str]:
    """Each of `holdout`'s scores' value, as the scores CSV renders it, by key."""
    chosen = [s for s in scores if s.holdout == holdout]
    values = [line.rsplit(",", 1)[1] for line in render(chosen).splitlines()[1:]]
    keys = [(s.holdout, s.arm, s.source.value, s.measure.value, s.scope.value) for s in chosen]
    if len(set(keys)) != len(keys):
        msg = f"holdout {holdout}'s scores hold a key twice"
        raise IdentityError(msg)
    return dict(zip(keys, values, strict=True))


def folder_scores(folder: Folder) -> list[Score]:
    """The folder's `scores.csv`, once its sha256 is the one its `scores` line logs.

    Raises:
        OSError: If the file cannot be read.
        IdentityError: If the log holds no `scores` line or the file's sha256 is not its.
        ScoresFormatError: If the file is not a scores CSV.
    """
    raw = (folder.path / SCORES_CSV).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != folder.scores_sha256:
        msg = f"{folder.path / SCORES_CSV}'s sha256 {sha} is not its scores line's"
        msg += f" {folder.scores_sha256}"
        raise IdentityError(msg)
    return parse(raw.decode("utf-8"))


def store(
    folder: Folder,
    path: Path,
    *,
    run: str,
    experiment: Experiment,
    data: Data,
    arms: Mapping[str, Runner],
    models: Mapping[Model, Factory],
    uci: bool,
) -> str:
    """Rescore `folder`, store every checked holdout's fills at `path`; return the scores CSV.

    Raises:
        FileExistsError, FileNotFoundError: As `creatable`.
        IdentityError: As `folder_scores`, `rescore`, `compare_holdout` and `RowsFile.write`.
        EvaluationError, BaselineError: As `rescore`.
    """
    theirs = folder_scores(folder)
    header = {
        "kind": "header",
        "schema": SCHEMA,
        "run": run,
        "folder": folder.path.name,
        "log_sha256": folder.log_sha256,
        "data": data.digest,
        "frame": data.frame.name,
    }
    with RowsFile(path, header) as rows:
        logger.info(
            "rows folder=%s run=%s holdouts=%d started_utc=%s pid=%d",
            folder.path,
            run,
            len(experiment.holdout_seeds),
            datetime.now(UTC).isoformat(timespec="seconds"),
            os.getpid(),
        )

        def after(holdout: int, logged: Logged, scored: Sequence[Score]) -> None:
            sampled = subsampled(logged.split)
            differing = compare_holdout(
                scored, theirs, holdout=holdout, admitted=held_out, sampled=sampled
            )
            count = rows.write(holdout, logged.fills, arms=folder.arms)
            logger.info(
                "rows holdout=%d stored=%d differing=%d subsampled=%s",
                holdout,
                count,
                differing,
                "true" if sampled else "false",
            )

        text = rescore(
            folder,
            [],
            run=run,
            experiment=experiment,
            data=data,
            arms=arms,
            models=models,
            uci=uci,
            after=after,
        )
        rows.open()
    whole = hashlib.sha256(path.read_bytes()).hexdigest()
    logger.info("rows done holdouts=%d stored=%d sha256=%s", rows.holdouts, rows.stored, whole)
    return text


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `rows` subcommand to `commands`."""
    storing = commands.add_parser(
        "rows", help="rescore a run's folder and store every fill its checks pass"
    )
    storing.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    storing.add_argument("--folder", type=Path, required=True, help="the run's folder under runs/")
    storing.add_argument(
        "--rows", type=Path, required=True, help="the store to create; it must not exist"
    )
    storing.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
