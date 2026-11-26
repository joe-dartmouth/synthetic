"""Read the members of a verified zip or gzipped tar archive as text, capped in size.

`member_text` reads one named zip member: `bko.datasets.diabetes` reads Diabetes 130's two members
through it. `members` and `tar_members` read every member of a zip or a gzipped tar whose name a
pattern matches, in archive order, from memory, never extracting to disk:
`bko.datasets.physionet2012` reads the PhysioNet 2012 sets through them. Each refuses a link, a
device, an absolute or parent path, a name outside its pattern, a repeated name, a member over
`MEMBER_LIMIT`, more than `TOTAL_LIMIT` bytes in all and a damaged archive.
"""

from __future__ import annotations

import io
import lzma
import re
import stat
import tarfile
import zipfile
import zlib
from collections.abc import Iterator
from typing import IO, Final, cast

from bko.datasets.errors import ContractError
from bko.datasets.sources import Source


MEMBER_LIMIT: Final = 64 * 2**20
"""The most bytes a member may hold once decompressed; the archive's largest holds under 20 MiB."""

TOTAL_LIMIT: Final = 256 * 2**20
"""The most bytes `members` and `tar_members` read from one archive's members in all."""

_ZIP_ERRORS: Final = (
    zipfile.BadZipFile,
    zlib.error,
    UnicodeDecodeError,
    NotImplementedError,
    RuntimeError,
    EOFError,
    OSError,
    lzma.LZMAError,
)
"""What reading a damaged, encrypted or non-UTF-8 zip raises: a corrupt bzip2 member raises
`OSError`, a corrupt LZMA member `lzma.LZMAError`."""

_TAR_ERRORS: Final = (tarfile.TarError, zlib.error, UnicodeDecodeError, EOFError, OSError)
"""What reading a damaged gzip stream, tar or non-UTF-8 member raises."""


def member_text(data: bytes, member: str, source: Source) -> str:
    """Return `member` of the zip `data` as UTF-8 text, refusing one over `MEMBER_LIMIT` bytes.

    Raises:
        ContractError: If the archive has no such member, the member holds more than
            `MEMBER_LIMIT` bytes, or it cannot be read as UTF-8 text.
    """
    where = f"{source.name}: cannot read {member}"
    try:
        return _capped(data, member, where).decode("utf-8")
    except KeyError as exc:
        msg = f"{where}: the archive has no such member"
        raise ContractError(msg) from exc
    except _ZIP_ERRORS as exc:
        msg = f"{where}: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc


def _capped(data: bytes, member: str, where: str) -> bytes:
    """Return `member`'s bytes, refusing one whose stated or read size passes `MEMBER_LIMIT`."""
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        size = bundle.getinfo(member).file_size
        if size > MEMBER_LIMIT:
            msg = f"{where}: {size} bytes, over the {MEMBER_LIMIT}-byte cap"
            raise ContractError(msg)
        with bundle.open(member) as handle:
            body = handle.read(MEMBER_LIMIT + 1)
    if len(body) > MEMBER_LIMIT:
        msg = f"{where}: over the {MEMBER_LIMIT}-byte cap"
        raise ContractError(msg)
    return body


class _Tally:
    """The names seen and the bytes read so far from one archive, refusing what breaks the caps."""

    def __init__(self, pattern: re.Pattern[str], source: Source) -> None:
        self.pattern = pattern
        self.source = source
        self.names: set[str] = set()
        self.total = 0

    def refuse(self, name: str, problem: str) -> ContractError:
        """The error naming the source, the member and its problem."""
        return ContractError(f"{self.source.name}: {name}: {problem}")

    def safe(self, name: str) -> None:
        """Refuse an absolute name or one with a parent step."""
        if name.startswith("/") or ".." in name.split("/"):
            raise self.refuse(name, "an absolute or parent path")

    def admit(self, name: str, size: int) -> None:
        """Refuse a file member by its name and stated size, before it is read."""
        self.safe(name)
        if not self.pattern.fullmatch(name):
            raise self.refuse(name, f"outside the pattern {self.pattern.pattern}")
        if name in self.names:
            raise self.refuse(name, "a repeated name")
        self.names.add(name)
        self.count(name, size)

    def count(self, name: str, size: int) -> None:
        """Refuse a member over `MEMBER_LIMIT` bytes, or one that takes the total over its cap."""
        if size > MEMBER_LIMIT:
            raise self.refuse(name, f"{size} bytes, over the {MEMBER_LIMIT}-byte cap")
        self.total += size
        if self.total > TOTAL_LIMIT:
            msg = f"{self.total} bytes in all, over the {TOTAL_LIMIT}-byte cap"
            raise self.refuse(name, msg)

    def text(self, name: str, body: bytes) -> str:
        """The member's bytes as UTF-8 text, refusing more than `MEMBER_LIMIT` read."""
        if len(body) > MEMBER_LIMIT:
            raise self.refuse(name, f"over the {MEMBER_LIMIT}-byte cap")
        return body.decode("utf-8")


def members(data: bytes, pattern: re.Pattern[str], source: Source) -> Iterator[tuple[str, str]]:
    """Yield each file member of the zip `data` and its UTF-8 text, in archive order.

    The archive is opened once and every entry is checked before any member is read; a
    directory entry is skipped.

    Raises:
        ContractError: If an entry has an absolute or parent path, a file member is not a regular
            file, its name does not fully match `pattern` or repeats, it states more than
            `MEMBER_LIMIT` bytes, the members state more than `TOTAL_LIMIT` in all, or the archive
            cannot be read or a member is not UTF-8 text.
    """
    try:
        yield from _zip_members(data, _Tally(pattern, source))
    except _ZIP_ERRORS as exc:
        msg = f"{source.name}: cannot read the archive: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc


def _zip_members(data: bytes, tally: _Tally) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        files = []
        for info in bundle.infolist():
            tally.safe(info.filename)
            if info.is_dir():
                continue
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind and kind != stat.S_IFREG:
                raise tally.refuse(info.filename, "not a regular file")
            tally.admit(info.filename, info.file_size)
            files.append(info)
        for info in files:
            with bundle.open(info) as handle:
                body = handle.read(MEMBER_LIMIT + 1)
            yield info.filename, tally.text(info.filename, body)


def tar_members(data: bytes, pattern: re.Pattern[str], source: Source) -> Iterator[tuple[str, str]]:
    """Yield each file member of the gzipped tar `data` and its UTF-8 text, in archive order.

    Each member is checked from its header before it is read, through `extractfile` only; a
    directory entry is skipped.

    Raises:
        ContractError: As `members`, and for a member that is a link, a device or any other entry
            but a regular file or a directory.
    """
    try:
        yield from _tar_members(data, _Tally(pattern, source))
    except _TAR_ERRORS as exc:
        msg = f"{source.name}: cannot read the archive: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc


def _tar_members(data: bytes, tally: _Tally) -> Iterator[tuple[str, str]]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        for member in bundle:
            tally.safe(member.name)
            if member.isdir():
                continue
            if not member.isreg():
                raise tally.refuse(member.name, "not a regular file: a link or a device")
            tally.admit(member.name, member.size)
            # A regular member always has a file object; only other kinds read as None.
            with cast("IO[bytes]", bundle.extractfile(member)) as handle:
                body = handle.read(MEMBER_LIMIT + 1)
            yield member.name, tally.text(member.name, body)
