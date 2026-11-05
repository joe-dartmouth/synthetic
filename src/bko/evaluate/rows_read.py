"""Read back a `rows` store: `read_rows`, `bind` and `frame_of`.

`read_rows` reads a store written by `bko.evaluate.rows`, `bind` ties it to its folder, frame and
every `arm` line, and `frame_of` turns one fill line into the frame its `arm` line hashes. ADR
`research/rebuilt-rows-store` holds the store's format and what a reader checks.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.evaluate.data import digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.run_folder import Folder, fields


SCHEMA: Final = 1
"""The store's schema, which its header names."""


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


def read_rows(path: Path) -> Rows:
    """Read the store at `path`.

    Raises:
        OSError: If it cannot be read.
        IdentityError: If a line is not a JSON object, its first line is not a schema-1 header, a
            later line's kind is not `rows`, it lacks a field or holds one of the wrong kind, or
            two lines name one holdout and arm.
    """
    text = path.read_text(encoding="utf-8")
    raw: list[dict[str, object]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            found = json.loads(line)
        except ValueError, RecursionError:
            found = None
        if not isinstance(found, dict):
            msg = f"line {number} of {path} is not a JSON object"
            raise IdentityError(msg)
        raw.append(found)
    schema = raw[0].get("schema") if raw else None
    if not raw or raw[0].get("kind") != "header" or type(schema) is not int or schema != SCHEMA:
        msg = f"{path} does not open with a schema-{SCHEMA} header line"
        raise IdentityError(msg)
    lines: dict[tuple[int, str], RowsLine] = {}
    for number, found in enumerate(raw[1:], start=2):
        line = _fill(found, f"line {number} of {path}")
        key = (line.holdout, line.arm)
        if key in lines:
            msg = f"{path} holds two lines for {key}"
            raise IdentityError(msg)
        lines[key] = line
    return Rows(raw[0], lines)


def _fill(found: Mapping[str, object], where: str) -> RowsLine:
    """The fill line `found`, read at `where`, once its kind and every field's are checked.

    Raises:
        IdentityError: If its kind is not `rows`, it lacks a field, a count is not a non-negative
            integer, or a text field is not text.
    """
    if found.get("kind") != "rows":
        msg = f"{where} is of kind {found.get('kind')!r}, not rows"
        raise IdentityError(msg)
    missing = [name for name in _FIELDS if name not in found]
    if missing:
        msg = f"{where} lacks {missing}"
        raise IdentityError(msg)
    return RowsLine(
        _count(found, "holdout", where),
        _text(found, "arm", where),
        _count(found, "attempt", where),
        _count(found, "added", where),
        _text(found, "sha256", where),
        _text(found, "csv", where),
    )


_FIELDS: Final = ("holdout", "arm", "attempt", "added", "sha256", "csv")


def _count(found: Mapping[str, object], name: str, where: str) -> int:
    """`found[name]`, refused unless a non-negative integer (a JSON `true` is not one)."""
    value = found[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{where} holds {name}={value!r}, not a count"
        raise IdentityError(msg)
    return value


def _text(found: Mapping[str, object], name: str, where: str) -> str:
    """`found[name]`, refused unless text."""
    value = found[name]
    if not isinstance(value, str):
        msg = f"{where} holds {name}={value!r}, not text"
        raise IdentityError(msg)
    return value


def bind(rows: Rows, folder: Folder, frame: FrameSpec, *, run: str, data: str) -> None:
    """Refuse a store written from another run, data, folder or frame, or one missing a fill.

    A store whose run stopped early holds a valid header and only the holdouts checked before the
    stop, so `bind` requires one fill line for each of the folder's `arm` lines.

    Args:
        rows: The store read back.
        folder: The folder it was written from.
        frame: The frame its rows read at.
        run: The named run it rescored.
        data: The digest of the data it rescored (`bko.evaluate.data.Data.digest`).

    Raises:
        IdentityError: If the header's run, data or frame is not the one given, its folder name or
            `log_sha256` is not `folder`'s, or its fill lines are not one for every `arm` line of
            `folder`.
    """
    for name, expected in (("run", run), ("data", data), ("frame", frame.name)):
        if rows.header.get(name) != expected:
            msg = f"the store names the {name} {rows.header.get(name)!r}, not {expected!r}"
            raise IdentityError(msg)
    named, logged = rows.header.get("folder"), rows.header.get("log_sha256")
    if named != folder.path.name:
        msg = f"the store names the folder {named!r}, not {folder.path.name!r}"
        raise IdentityError(msg)
    if logged != folder.log_sha256:
        msg = f"the store's log_sha256 {logged} is not {folder.path}'s {folder.log_sha256}"
        raise IdentityError(msg)
    if set(rows.lines) != set(folder.arms):
        missing, extra = (
            sorted(set(folder.arms) - set(rows.lines)),
            sorted(set(rows.lines) - set(folder.arms)),
        )
        msg = f"the store's fill lines are not one for every arm line of {folder.path}:"
        msg += f" missing {missing}, extra {extra}"
        raise IdentityError(msg)


def frame_of(line: RowsLine, *, arm_line: str, frame: FrameSpec) -> pd.DataFrame:
    """The rows `line` stores, checked against its sha256 and the folder's `arm_line`.

    Raises:
        IdentityError: If `line`'s holdout, arm, attempt, rows or sha256 is not `arm_line`'s, its
            CSV does not hash to its sha256 or read at `frame`'s dtypes, its columns are not
            `frame`'s, or the frame read back does not hash to its sha256.
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
    try:
        read = pd.read_csv(
            io.StringIO(line.csv),
            index_col="record_id",
            dtype=dtypes,
            float_precision="round_trip",
            keep_default_na=False,
            na_values=[""],
        )
    except (ValueError, OverflowError) as exc:
        msg = f"holdout {line.holdout}'s {line.arm} csv does not read as {frame.name}: {exc}"
        raise IdentityError(msg) from exc
    if list(read.columns) != list(frame.columns):
        msg = f"holdout {line.holdout}'s {line.arm} columns {list(read.columns)} are not"
        msg += f" {frame.name}'s"
        raise IdentityError(msg)
    if len(read) != line.added or digest(read) != line.sha256:
        msg = f"holdout {line.holdout}'s {line.arm} rows read back to another sha256"
        msg += f" than {line.sha256}"
        raise IdentityError(msg)
    return read
