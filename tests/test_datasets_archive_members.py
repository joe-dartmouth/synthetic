"""`bko.datasets.archive.members` and `tar_members`: every member of a zip or gzipped tar, checked.

Each archive is built here in memory; nothing is extracted to disk.
"""

from __future__ import annotations

import contextlib
import io
import re
import stat
import tarfile
import zipfile
from collections.abc import Callable, Iterator

import pytest

from bko.datasets.archive import members, tar_members
from bko.datasets.errors import ContractError
from bko.datasets.sources import Source
from tests._datasets_fixtures import pin


PATTERN = re.compile(r"set-a/\d+\.txt")
SOURCE = pin(b"archive", "physionet2012-set-a")

Reader = Callable[[bytes, re.Pattern[str], Source], Iterator[tuple[str, str]]]


def zipped(*entries: tuple[str, str]) -> bytes:
    """A zip holding each `(name, text)`, a name ending in `/` written as a directory entry."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, text in entries:
            bundle.writestr(zipfile.ZipInfo(name, date_time=(2012, 5, 8, 1, 23, 0)), text)
    return buffer.getvalue()


def tarred(*entries: tarfile.TarInfo | tuple[str, str]) -> bytes:
    """A gzipped tar holding each `(name, text)` as a regular file, a name ending in `/` as a
    directory entry, or each `TarInfo` as it is."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for entry in entries:
            if isinstance(entry, tarfile.TarInfo):
                bundle.addfile(entry)
                continue
            name, text = entry
            if name.endswith("/"):
                bundle.addfile(directory(name))
                continue
            body = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def directory(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info


def special(name: str, kind: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = "set-a/1.txt"
    return info


def build(kind: str, *entries: tuple[str, str]) -> tuple[bytes, Reader]:
    """The archive of `entries` in `kind`, `zip` or `tar`, and the reader for it."""
    if kind == "zip":
        return zipped(*entries), members
    return tarred(*entries), tar_members


KINDS = pytest.mark.parametrize("kind", ["zip", "tar"])


@KINDS
def test_members_yields_each_matching_file_once(kind: str) -> None:
    data, read = build(kind, ("set-a/", ""), ("set-a/2.txt", "b\n"), ("set-a/1.txt", "a\n"))
    assert list(read(data, PATTERN, SOURCE)) == [("set-a/2.txt", "b\n"), ("set-a/1.txt", "a\n")]


def test_a_tar_directory_entry_is_skipped() -> None:
    data = tarred(directory("set-a"), ("set-a/1.txt", "a\n"))
    assert list(tar_members(data, PATTERN, SOURCE)) == [("set-a/1.txt", "a\n")]


@KINDS
@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("../x", "an absolute or parent path"),
        ("set-a/../set-a/1.txt", "an absolute or parent path"),
        ("/set-a/1.txt", "an absolute or parent path"),
        ("set-b/1.txt", "outside the pattern"),
        ("set-a/1.txt.bak", "outside the pattern"),
    ],
)
def test_a_name_outside_the_pattern_is_refused(kind: str, name: str, match: str) -> None:
    data, read = build(kind, ("set-a/1.txt", "a\n"), (name, "x\n"))
    with pytest.raises(ContractError, match=re.escape(f"physionet2012-set-a: {name}: {match}")):
        list(read(data, PATTERN, SOURCE))


@KINDS
def test_a_parent_path_directory_entry_is_refused(kind: str) -> None:
    data, read = build(kind, ("../set-a/", ""), ("set-a/1.txt", "a\n"))
    with pytest.raises(ContractError, match="an absolute or parent path"):
        list(read(data, PATTERN, SOURCE))


@KINDS
def test_a_repeated_name_is_refused(kind: str) -> None:
    warned = pytest.warns(UserWarning, match="Duplicate name")
    with warned if kind == "zip" else contextlib.nullcontext():
        data, read = build(kind, ("set-a/1.txt", "a\n"), ("set-a/1.txt", "b\n"))
    with pytest.raises(ContractError, match=r"set-a/1\.txt: a repeated name"):
        list(read(data, PATTERN, SOURCE))


@KINDS
def test_a_member_over_the_cap_is_refused_before_it_is_read(kind: str, monkeypatch) -> None:
    data, read = build(kind, ("set-a/1.txt", "a\n"), ("set-a/2.txt", "x" * 101))
    monkeypatch.setattr("bko.datasets.archive.MEMBER_LIMIT", 100)
    opened: list[str] = []
    real_zip_open, real_extract = zipfile.ZipFile.open, tarfile.TarFile.extractfile

    def zip_open(self, member, *args, **kwargs):
        opened.append(getattr(member, "filename", member))
        return real_zip_open(self, member, *args, **kwargs)

    def extract(self, member):
        opened.append(member.name)
        return real_extract(self, member)

    monkeypatch.setattr(zipfile.ZipFile, "open", zip_open)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", extract)
    with pytest.raises(ContractError, match=r"set-a/2.txt: 101 bytes, over the 100-byte cap"):
        list(read(data, PATTERN, SOURCE))
    assert "set-a/2.txt" not in opened


@KINDS
def test_a_total_over_the_cap_is_refused(kind: str, monkeypatch) -> None:
    data, read = build(kind, ("set-a/1.txt", "x" * 60), ("set-a/2.txt", "y" * 60))
    monkeypatch.setattr("bko.datasets.archive.TOTAL_LIMIT", 100)
    with pytest.raises(ContractError, match=r"120 bytes in all, over the 100-byte cap"):
        list(read(data, PATTERN, SOURCE))
    monkeypatch.setattr("bko.datasets.archive.TOTAL_LIMIT", 120)
    assert len(list(read(data, PATTERN, SOURCE))) == 2


@pytest.mark.parametrize(
    "kind",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE],
    ids=["symlink", "hard-link", "char-device", "block-device", "fifo"],
)
def test_a_tar_link_or_device_member_is_refused(kind: bytes) -> None:
    data = tarred(("set-a/1.txt", "a\n"), special("set-a/2.txt", kind))
    with pytest.raises(ContractError, match=r"set-a/2\.txt: not a regular file"):
        list(tar_members(data, PATTERN, SOURCE))


def test_a_zip_symlink_member_is_refused() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        info = zipfile.ZipInfo("set-a/2.txt", date_time=(2012, 5, 8, 1, 23, 0))
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "set-a/1.txt")
    with pytest.raises(ContractError, match=r"set-a/2\.txt: not a regular file"):
        list(members(buffer.getvalue(), PATTERN, SOURCE))


def test_a_zip_member_with_a_regular_files_mode_is_read() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        info = zipfile.ZipInfo("set-a/1.txt", date_time=(2012, 5, 8, 1, 23, 0))
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        bundle.writestr(info, "a\n")
    assert list(members(buffer.getvalue(), PATTERN, SOURCE)) == [("set-a/1.txt", "a\n")]


def _truncated(data: bytes) -> bytes:
    return data[: len(data) // 2]


def _latin1_tar() -> bytes:
    buffer = io.BytesIO()
    body = "é\n".encode("latin-1")
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("set-a/1.txt")
        info.size = len(body)
        bundle.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: b"not a gzip stream", r"physionet2012-set-a: cannot read the archive: \w+"),
        (
            lambda: _truncated(tarred(*((f"set-a/{n}.txt", f"{n}\n" * 400) for n in range(30)))),
            r"physionet2012-set-a: cannot read the archive: \w+",
        ),
        (
            lambda: tarred(("set-a/1.txt", "a\n")).replace(b"\x1f\x8b", b"\x1f\x8c", 1),
            r"physionet2012-set-a: cannot read the archive: \w+",
        ),
        (_latin1_tar, "UnicodeDecodeError"),
    ],
    ids=["not-a-gzip", "truncated", "bad-magic", "not-utf-8"],
)
def test_a_damaged_tar_is_a_contract_error(make, match: str) -> None:
    with pytest.raises(ContractError, match=match):
        list(tar_members(make(), PATTERN, SOURCE))


def _bad_crc(data: bytes, name: str) -> bytes:
    """Flip the first stored byte after `name`'s local header, so its CRC no longer matches."""
    start = data.index(name.encode()) + len(name)
    return data[:start] + bytes([data[start] ^ 0xFF]) + data[start + 1 :]


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: b"not a zip archive", "BadZipFile"),
        (lambda: _truncated(zipped(("set-a/1.txt", "a\n" * 2000))), "BadZipFile"),
        (lambda: _bad_crc(zipped(("set-a/1.txt", "a\n" * 2000)), "set-a/1.txt"), r"\w+"),
    ],
    ids=["not-a-zip", "truncated", "bad-crc"],
)
def test_a_damaged_zip_is_a_contract_error(make, match: str) -> None:
    with pytest.raises(
        ContractError, match=rf"physionet2012-set-a: cannot read the archive: {match}"
    ):
        list(members(make(), PATTERN, SOURCE))
