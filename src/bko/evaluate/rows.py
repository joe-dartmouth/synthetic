"""`python -m bko.evaluate rows`: rescore a run's folder and store every fill it checks.

`store` rescores a recorded folder through `bko.evaluate.shared.rescore`, at the folder's own
models, and makes no API call. After a holdout passes identity checks 3 and 4, `compare_holdout`
holds it to the strict check 7 against the folder's `scores.csv`; only then does `RowsFile.write`
append every fill of that holdout to the store, so a holdout that fails a check (exit 3) stores
nothing. ADR `research/rebuilt-rows-store` holds the store's format and what a reader checks;
ADR `research/shared-test-split-reading` holds the strict check 7 and its 50,000-row exception.

`read_rows` reads a store back, `bind` ties it to its folder, and `frame_of` turns one fill line
into the frame its `arm` line hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Self, TextIO

import numpy as np
import pandas as pd

from bko.baselines.methods import Runner
from bko.datasets.frames import FrameSpec
from bko.evaluate.classifier import Factory, Model
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import Experiment
from bko.evaluate.filling import Filled, Logged
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.run_folder import Folder, fields
from bko.evaluate.scores import Measure, Score, Source, parse, render
from bko.evaluate.shared import rescore


logger = logging.getLogger("bko.evaluate")

SCHEMA: Final = 1
"""The store's schema, which its header names."""

SCORES_CSV: Final = "scores.csv"

SUBSAMPLED_ABOVE: Final = 50_000
"""The training rows above which a folder's replica fidelity may hold one subsample draw."""

SUBSAMPLED_MEASURES: Final = frozenset({Measure.QUALITY, Measure.COLUMN_PAIR_TRENDS})
"""The replica measures a subsampled holdout's rescore may read differently."""

_TRAIN: Final = re.compile(r" train=(?P<train>\d+) ")

type ScoreKey = tuple[int, str, str, str, str]
"""A score's holdout, arm, source, measure and scope, as the scores CSV writes them."""


@dataclass(frozen=True, slots=True)
class RowsLine:
    """One stored fill.

    Attributes:
        holdout: The holdout's seed.
        arm: The `arm` line's arm: `<arm>`, or `<arm>-replica` for a replica fill.
        attempt: The attempt that returned.
        added: The rows it added.
        sha256: The sha256 of `csv`, which the `arm` line logs.
        csv: The rows as `bko.evaluate.data.digest` writes them.
    """

    holdout: int
    arm: str
    attempt: int
    added: int
    sha256: str
    csv: str


@dataclass(frozen=True, slots=True)
class Rows:
    """A store read back: its header and each fill line, by holdout and arm."""

    header: Mapping[str, object]
    lines: Mapping[tuple[int, str], RowsLine]


class RowsFile:
    """A store being written: created with mode 0600 and `O_EXCL`, its header first."""

    def __init__(self, path: Path, header: Mapping[str, object]) -> None:
        """Create the store at `path` and write `header`.

        Raises:
            FileExistsError: If `path` exists.
        """
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file: TextIO = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._put(header)
        self._file.flush()

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

    def write(self, holdout: int, fills: Sequence[Filled]) -> int:
        """Append every fill of `holdout`, then flush; return how many.

        Every line is built before any is written, so a refused fill stores none of the holdout.

        Raises:
            IdentityError: If a fill holds a non-finite value, or its rows' sha256 or count is
                not its `arm` line's.
        """
        lines = [_line(holdout, fill) for fill in fills]
        for line in lines:
            self._put(line)
            logger.info(
                "rows holdout=%d arm=%s added=%d sha256=%s matches_arm=true",
                holdout,
                line["arm"],
                line["added"],
                line["sha256"],
            )
        self._file.flush()
        return len(lines)

    def _put(self, line: Mapping[str, object]) -> None:
        self._file.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")


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


def read_rows(path: Path) -> Rows:
    """Read the store at `path`.

    Raises:
        OSError: If it cannot be read.
        IdentityError: If its first line is not a schema-1 header, a later line is not a fill
            line, or two lines name one holdout and arm.
    """
    text = path.read_text(encoding="utf-8")
    raw = [json.loads(line) for line in text.splitlines()]
    if not raw or raw[0].get("kind") != "header" or raw[0].get("schema") != SCHEMA:
        msg = f"{path} does not open with a schema-{SCHEMA} header line"
        raise IdentityError(msg)
    lines: dict[tuple[int, str], RowsLine] = {}
    for found in raw[1:]:
        try:
            line = RowsLine(
                int(found["holdout"]),
                str(found["arm"]),
                int(found["attempt"]),
                int(found["added"]),
                str(found["sha256"]),
                str(found["csv"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"{path} holds a line that is not a fill line: {exc}"
            raise IdentityError(msg) from exc
        key = (line.holdout, line.arm)
        if found.get("kind") != "rows" or key in lines:
            msg = f"{path} holds two lines for {key}, or one whose kind is not rows"
            raise IdentityError(msg)
        lines[key] = line
    return Rows(raw[0], lines)


def bind(rows: Rows, folder: Folder) -> None:
    """Refuse a store written from another folder than `folder`.

    Raises:
        IdentityError: If the header's folder name or `log_sha256` is not `folder`'s.
    """
    named, logged = rows.header.get("folder"), rows.header.get("log_sha256")
    if named != folder.path.name:
        msg = f"the store names the folder {named!r}, not {folder.path.name!r}"
        raise IdentityError(msg)
    if logged != folder.log_sha256:
        msg = f"the store's log_sha256 {logged} is not {folder.path}'s {folder.log_sha256}"
        raise IdentityError(msg)


def frame_of(line: RowsLine, *, arm_line: str, frame: FrameSpec) -> pd.DataFrame:
    """The rows `line` stores, checked against its sha256 and the folder's `arm_line`.

    Raises:
        IdentityError: If `line`'s holdout, arm, attempt, rows or sha256 is not `arm_line`'s, its
            CSV does not hash to its sha256, or the frame read back does not.
    """
    logged = fields(arm_line).group("holdout", "arm", "attempt", "added", "sha256")
    ours = (str(line.holdout), line.arm, str(line.attempt), str(line.added), line.sha256)
    if ours != logged:
        msg = f"the stored fill {ours} is not its arm line {arm_line!r}"
        raise IdentityError(msg)
    if hashlib.sha256(line.csv.encode()).hexdigest() != line.sha256:
        msg = f"holdout {line.holdout}'s {line.arm} csv does not hash to its sha256 {line.sha256}"
        raise IdentityError(msg)
    dtypes: dict[Hashable, str] = {**frame.dtypes}
    read = pd.read_csv(
        io.StringIO(line.csv),
        index_col="record_id",
        dtype=dtypes,
        float_precision="round_trip",
    )
    if len(read) != line.added or digest(read) != line.sha256:
        msg = f"holdout {line.holdout}'s {line.arm} rows read back to another sha256"
        msg += f" than {line.sha256}"
        raise IdentityError(msg)
    return read


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
            found = None if logged.split is None else _TRAIN.search(logged.split)
            if found is None:
                msg = f"holdout {holdout}'s split line {logged.split!r} names no training rows"
                raise IdentityError(msg)
            subsampled = int(found["train"]) > SUBSAMPLED_ABOVE
            differing = compare_holdout(
                scored, theirs, holdout=holdout, admitted=held_out, subsampled=subsampled
            )
            count = rows.write(holdout, logged.fills)
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
