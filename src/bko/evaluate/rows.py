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
from bko.evaluate.data import Data
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


class RowsFile:
    """A store being written: created with mode 0600 and `O_EXCL`, its header first."""

    def __init__(self, path: Path, header: Mapping[str, object]) -> None:
        """Create the store at `path` and write `header`.

        Raises:
            FileExistsError: If `path` exists.
        """
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file: TextIO = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._written: set[tuple[int, str]] = set()
        self._put(header)
        self._sync()

    def __enter__(self) -> Self:
        """The open store."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the store."""
        self._file.close()

    def write(
        self, holdout: int, fills: Sequence[Filled], *, arms: Mapping[tuple[int, str], str]
    ) -> int:
        """Append every fill of `holdout`, then flush and sync; return how many.

        Every line is built before any is written, so a refused fill stores none of the holdout.

        Args:
            holdout: The holdout's seed.
            fills: Its fills.
            arms: The folder's `arm` lines by holdout and arm; each fill's line must be its own.

        Raises:
            IdentityError: If a fill holds a non-finite value, its rows' sha256 or count is not its
                `arm` line's, its `arm` line is not the folder's, or its holdout and arm are
                already stored.
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
        self._written |= set(keys)
        for line in lines:
            self._put(line)
            logger.info(
                "rows holdout=%d arm=%s added=%d sha256=%s matches_arm=true",
                holdout,
                line["arm"],
                line["added"],
                line["sha256"],
            )
        self._sync()
        return len(lines)

    def _put(self, line: Mapping[str, object]) -> None:
        self._file.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")

    def _sync(self) -> None:
        """Flush the store and sync it to disk, so a power loss keeps each written holdout."""
        self._file.flush()
        os.fsync(self._file.fileno())


def _line(holdout: int, fill: Filled) -> dict[str, object]:
    """`fill`'s store line, checked against its `arm` line."""
    added = fill.baseline.added
    numbers = added.select_dtypes("number").to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        msg = f"holdout {holdout}'s {fill.name} rows hold a non-finite value"
        raise IdentityError(msg)
    csv = added.to_csv(lineterminator="\n")
    sha = hashlib.sha256(csv.encode()).hexdigest()
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
    subsampled: bool,
) -> int:
    """The strict check 7 on one holdout; return how many replica fidelity rows differ.

    Each key's rendered line is compared, so NaN matches NaN. A key only `ours` holds must be
    `admitted`; a replica's `quality` and `column-pair-trends` may differ when `subsampled`.

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
        if subsampled and loose:
            differing += 1
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
        FileExistsError: If `path` exists.
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
            differing = compare_holdout(
                scored,
                theirs,
                holdout=holdout,
                admitted=held_out,
                subsampled=subsampled(logged.split),
            )
            count = rows.write(holdout, logged.fills, arms=folder.arms)
            logger.info("rows holdout=%d stored=%d differing=%d", holdout, count, differing)

        return rescore(
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
