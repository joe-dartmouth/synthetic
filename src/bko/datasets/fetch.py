"""Download a pinned source into the data root once, and verify it whenever it is used.

A snapshot is the directory `<data root>/<source name>/<first 16 hex digits of its sha256>/`,
holding the download and `source.json`. The directory appears only through a rename after the bytes
match their pin, so a snapshot directory that exists is complete. `source.json` records the URL,
size, sha256 and download time an experiment cites in `docs/EVIDENCE.md`; verification reads the
pin in `bko.datasets.sources`, never the record.

Only `ensure` touches the network, and it replaces a snapshot that fails its pin. Loaders call
`read_verified`, which never fetches and refuses such a snapshot. A reader that must open the file
itself, as pronto must to honour an encoding, calls `verified_path`, which verifies the bytes and
returns the path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from http.client import HTTPException, HTTPMessage
from pathlib import Path
from typing import IO, BinaryIO, Final

from bko.datasets.errors import ContractError, DatasetError, UnavailableError
from bko.datasets.sources import Source


logger = logging.getLogger(__name__)

CHUNK_BYTES: Final = 1 << 20
"""Bytes read per call, from the network or from disk."""

TIMEOUT_SECONDS: Final = 60.0
"""The per-operation socket timeout of a download, in seconds."""

DEADLINE_SECONDS: Final = 600.0
"""The longest a whole download may take, in seconds, checked after each chunk."""

RECORD_NAME: Final = "source.json"
"""The file in a snapshot directory that records where and when its bytes came from."""

type UrlOpener = Callable[[str, float], BinaryIO]
"""Open an HTTPS URL with a per-operation timeout in seconds and return the response body."""


class HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another HTTPS address."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Refuse a redirect to any scheme but HTTPS, and otherwise follow it as urllib does.

        Raises:
            UnavailableError: If `newurl` is not an HTTPS address.
        """
        if not newurl.startswith("https://"):
            text = f"refusing a redirect from {req.full_url} to {newurl}: only https:// is fetched"
            raise UnavailableError(text)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER: Final = urllib.request.build_opener(HttpsOnlyRedirects)


def open_https(url: str, timeout: float) -> BinaryIO:
    """Open `url` for reading, refusing every scheme but HTTPS, redirects included.

    Args:
        url: The address to open.
        timeout: The per-operation socket timeout in seconds.

    Returns:
        The response body.

    Raises:
        UnavailableError: If `url`, or a redirect it answers with, is not an HTTPS address.
    """
    if not url.startswith("https://"):
        msg = f"refusing {url}: only https:// sources are fetched"
        raise UnavailableError(msg)
    return _OPENER.open(url, timeout=timeout)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def snapshot_dir(source: Source, data_dir: Path) -> Path:
    """Return the directory that holds `source`'s snapshot under `data_dir`."""
    return data_dir / source.name / source.sha256[:16]


_THIS_FILE: Final = Path(__file__)


def default_data_dir(anchor: Path = _THIS_FILE) -> Path:
    """Return `data/` at the root of the checkout this module runs from.

    The editable install runs this file from `src/bko/datasets/`, three levels below the root,
    whose `.gitignore` ignores `/data/`. Elsewhere, as in an installed wheel, there is no default.

    Raises:
        UnavailableError: If `anchor` does not sit in a checkout's `src/bko/datasets/`.
    """
    parents = anchor.resolve().parents
    root = parents[3] if len(parents) > 3 else None
    if (
        root is None
        or not (root / "pyproject.toml").is_file()
        or not (root / "src" / "bko").is_dir()
    ):
        msg = f"no repository root above {anchor}; pass --data-dir"
        raise UnavailableError(msg)
    return root / "data"


def verify(path: Path, source: Source) -> None:
    """Check the file at `path` against `source`'s size and sha256, reading it in chunks.

    The size and the sha256 are taken from the same bytes.

    Raises:
        UnavailableError: If the file is absent.
        ContractError: If its size or sha256 differs from the pin.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                size += len(chunk)
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise _no_snapshot(source, path) from exc
    _match(source, size, digest.hexdigest(), where=str(path))


def read_verified(source: Source, data_dir: Path) -> bytes:
    """Return the bytes of `source`'s snapshot under `data_dir` once they match the pin.

    The bytes are hashed as read, so the bytes returned are the bytes checked.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If its size or sha256 differs from the pin.
    """
    path = snapshot_dir(source, data_dir) / source.filename
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise _no_snapshot(source, path) from exc
    _match(source, len(data), hashlib.sha256(data).hexdigest(), where=str(path))
    return data


def verified_path(source: Source, data_dir: Path) -> Path:
    """Return the path of `source`'s snapshot under `data_dir` once its bytes match the pin.

    For a reader that must open the file itself. The reader reads the file again after this check,
    so only an out-of-band edit between the two could change what it reads. It never fetches.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If its size or sha256 differs from the pin.
    """
    path = snapshot_dir(source, data_dir) / source.filename
    verify(path, source)
    return path


def ensure(
    source: Source,
    data_dir: Path,
    *,
    open_url: UrlOpener = open_https,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    """Make `source`'s snapshot present and verified under `data_dir`, and return its file.

    A verified snapshot is used as it is, with no request. One that fails its pin or its record is
    removed and fetched again. A download is written into a staging directory beside the snapshot
    and renamed into place only once it matches the pin. The snapshot's identity is logged.

    Args:
        source: The pinned download.
        data_dir: The data root.
        open_url: The HTTPS opener.
        now: The clock for the record's download time; it must return an aware UTC datetime.

    Returns:
        The path of the verified file.

    Raises:
        UnavailableError: If the download cannot be completed.
        ContractError: If the downloaded bytes differ from the pin.
        OSError: If the data root cannot be written.
    """
    final = snapshot_dir(source, data_dir)
    path = final / source.filename
    if final.is_dir():
        try:
            verify(path, source)
            retrieved = _retrieved_at(source, data_dir)
        except (DatasetError, FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "replacing snapshot %s of %s, which fails its pin or record: %s: %s",
                final,
                source.name,
                type(exc).__name__,
                exc,
            )
            shutil.rmtree(final)
        else:
            _log_identity("snapshot present", source, path, retrieved)
            return path
    logger.info("fetching %s url=%s size=%d", source.name, source.url, source.size)
    staging = staging_beside(final)
    try:
        _download(source, staging / source.filename, open_url)
        retrieved = _write_record(staging / RECORD_NAME, source, now())
        _publish(staging, final, path, source)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _log_identity("fetched", source, path, retrieved)
    return path


def staging_beside(final: Path) -> Path:
    """Create an empty, hidden staging directory beside `final` and return it.

    Work is written there and renamed to `final` once complete, so `final` never holds a partial
    result. `mkdir` honours the umask, so the renamed directory gets the mode of those around it.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.{secrets.token_hex(8)}"
    staging.mkdir()
    return staging


def read_record(source: Source, data_dir: Path) -> dict[str, object]:
    """Return the `source.json` record of `source`'s snapshot under `data_dir`."""
    text = (snapshot_dir(source, data_dir) / RECORD_NAME).read_text(encoding="utf-8")
    return json.loads(text)


def _retrieved_at(source: Source, data_dir: Path) -> str:
    value = read_record(source, data_dir)["retrieved_at"]
    if not isinstance(value, str):
        msg = f"{RECORD_NAME} of {source.name}: retrieved_at is {value!r}, not a string"
        raise TypeError(msg)
    return value


def _log_identity(event: str, source: Source, path: Path, retrieved_at: str) -> None:
    logger.info(
        "%s %s path=%s sha256=%s retrieved_at=%s",
        event,
        source.name,
        path,
        source.sha256,
        retrieved_at,
    )


def _publish(staging: Path, final: Path, path: Path, source: Source) -> None:
    """Rename `staging` to `final`; if a concurrent run published first, verify its snapshot."""
    try:
        staging.rename(final)
    except OSError:
        if not final.is_dir():
            raise
        logger.info("snapshot %s was published by a concurrent run; verifying it", final)
        verify(path, source)


def _download(source: Source, dest: Path, open_url: UrlOpener) -> None:
    try:
        with open_url(source.url, TIMEOUT_SECONDS) as response, dest.open("wb") as out:
            size, sha256 = _copy(response, out, limit=source.size, where=source.url)
    except urllib.error.HTTPError as exc:
        exc.close()
        msg = f"{source.url} answered HTTP {exc.code} {exc.reason}"
        raise UnavailableError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"could not reach {source.url}: {exc.reason}"
        raise UnavailableError(msg) from exc
    except (TimeoutError, HTTPException) as exc:
        msg = f"download of {source.url} failed: {exc!r}"
        raise UnavailableError(msg) from exc
    except OSError as exc:
        msg = f"download of {source.url} into {dest} failed: {exc!r}"
        raise UnavailableError(msg) from exc
    _match(source, size, sha256, where=source.url)


def _copy(response: BinaryIO, out: BinaryIO, *, limit: int, where: str) -> tuple[int, str]:
    """Copy `response` to `out`, refusing more than `limit` bytes; return the size and sha256."""
    deadline = time.monotonic() + DEADLINE_SECONDS
    digest = hashlib.sha256()
    size = 0
    while chunk := response.read(CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            msg = f"{where}: more than the pinned {limit} bytes arrived"
            raise ContractError(msg)
        if time.monotonic() > deadline:
            msg = f"{where}: the download took longer than {DEADLINE_SECONDS:g} s"
            raise UnavailableError(msg)
        digest.update(chunk)
        out.write(chunk)
    return size, digest.hexdigest()


def _match(source: Source, size: int, sha256: str, *, where: str) -> None:
    if size != source.size or sha256 != source.sha256:
        msg = (
            f"{where}: {size} bytes with sha256 {sha256}; "
            f"pinned {source.size} bytes with sha256 {source.sha256}"
        )
        raise ContractError(msg)


def _no_snapshot(source: Source, path: Path) -> UnavailableError:
    return UnavailableError(f"no snapshot of {source.name} at {path}")


def _write_record(path: Path, source: Source, retrieved_at: datetime) -> str:
    """Write the snapshot's record and return its `retrieved_at`."""
    stamp = retrieved_at.isoformat()
    record = {
        "retrieved_at": stamp,
        "sha256": source.sha256,
        "size": source.size,
        "url": source.url,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stamp
