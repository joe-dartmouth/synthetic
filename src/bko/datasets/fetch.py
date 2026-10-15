"""Download a pinned source into the data root once, and verify it whenever it is used.

A snapshot is the directory `<data root>/<source name>/<first 16 hex digits of its sha256>/`,
holding the download and `source.json`. The directory appears only through a rename after the bytes
match their pin, so a snapshot directory that exists is complete. `source.json` records the URL,
size, sha256 and download time an experiment cites in `docs/EVIDENCE.md`; verification reads the
pin in `bko.datasets.sources`, never the record.

Only `ensure` touches the network. Loaders call `read_verified`, which never fetches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from typing import BinaryIO, Final

from bko.datasets.errors import ContractError, DatasetError, UnavailableError
from bko.datasets.sources import Source


logger = logging.getLogger(__name__)

CHUNK_BYTES: Final = 1 << 20
"""Bytes read from the network per call."""

TIMEOUT_SECONDS: Final = 60.0
"""The per-operation socket timeout of a download, in seconds."""

RECORD_NAME: Final = "source.json"
"""The file in a snapshot directory that records where and when its bytes came from."""

type UrlOpener = Callable[[str, float], BinaryIO]
"""Open an HTTPS URL with a per-operation timeout in seconds and return the response body."""


def open_https(url: str, timeout: float) -> BinaryIO:
    """Open `url` for reading, refusing every scheme but HTTPS.

    Args:
        url: The address to open.
        timeout: The per-operation socket timeout in seconds.

    Returns:
        The response body.

    Raises:
        UnavailableError: If `url` is not an HTTPS address.
    """
    if not url.startswith("https://"):
        msg = f"refusing {url}: only https:// sources are fetched"
        raise UnavailableError(msg)
    # The scheme is checked above, which is the audit this rule asks for.
    return urllib.request.urlopen(url, timeout=timeout)  # ruff: ignore[suspicious-url-open-usage]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def snapshot_dir(source: Source, data_dir: Path) -> Path:
    """Return the directory that holds `source`'s snapshot under `data_dir`."""
    return data_dir / source.name / source.sha256[:16]


def verify(path: Path, source: Source) -> None:
    """Check the file at `path` against `source`'s size and sha256, reading it in chunks.

    Raises:
        UnavailableError: If the file is absent.
        ContractError: If its size or sha256 differs from the pin.
    """
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256")
    except FileNotFoundError as exc:
        msg = f"no snapshot of {source.name} at {path}"
        raise UnavailableError(msg) from exc
    _match(source, path.stat().st_size, digest.hexdigest(), where=str(path))


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
        msg = f"no snapshot of {source.name} at {path}"
        raise UnavailableError(msg) from exc
    _match(source, len(data), hashlib.sha256(data).hexdigest(), where=str(path))
    return data


def ensure(
    source: Source,
    data_dir: Path,
    *,
    open_url: UrlOpener = open_https,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    """Make `source`'s snapshot present and verified under `data_dir`, and return its file.

    A verified snapshot is used as it is, with no request. One that fails its pin is removed and
    fetched again. A download is written into a staging directory beside the snapshot and renamed
    into place only once it matches the pin.

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
    """
    final = snapshot_dir(source, data_dir)
    path = final / source.filename
    if final.is_dir():
        try:
            verify(path, source)
            retrieved = read_record(source, data_dir)["retrieved_at"]
        except (DatasetError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("replacing a snapshot that fails its pin or record: %s", exc)
            shutil.rmtree(final)
        else:
            logger.info("snapshot present path=%s retrieved_at=%s", path, retrieved)
            return path
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".{final.name}."))
    try:
        _download(source, staging / source.filename, open_url)
        _write_record(staging / RECORD_NAME, source, now())
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info("fetched path=%s sha256=%s", path, source.sha256)
    return path


def read_record(source: Source, data_dir: Path) -> dict[str, object]:
    """Return the `source.json` record of `source`'s snapshot under `data_dir`."""
    text = (snapshot_dir(source, data_dir) / RECORD_NAME).read_text(encoding="utf-8")
    return json.loads(text)


def _download(source: Source, dest: Path, open_url: UrlOpener) -> None:
    try:
        with open_url(source.url, TIMEOUT_SECONDS) as response, dest.open("wb") as out:
            size, sha256 = _copy(response, out, limit=source.size)
    except urllib.error.HTTPError as exc:
        msg = f"{source.url} answered HTTP {exc.code} {exc.reason}"
        raise UnavailableError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"could not reach {source.url}: {exc.reason}"
        raise UnavailableError(msg) from exc
    except (TimeoutError, HTTPException) as exc:
        msg = f"download of {source.url} failed: {exc!r}"
        raise UnavailableError(msg) from exc
    _match(source, size, sha256, where=source.url)


def _copy(response: BinaryIO, out: BinaryIO, *, limit: int) -> tuple[int, str]:
    """Copy `response` to `out`, refusing more than `limit` bytes; return the size and sha256."""
    digest = hashlib.sha256()
    size = 0
    while chunk := response.read(CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            msg = f"more than the pinned {limit} bytes arrived"
            raise ContractError(msg)
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


def _write_record(path: Path, source: Source, retrieved_at: datetime) -> None:
    record = {
        "retrieved_at": retrieved_at.isoformat(),
        "sha256": source.sha256,
        "size": source.size,
        "url": source.url,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
