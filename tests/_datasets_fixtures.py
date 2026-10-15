"""Offline inputs for the `bko.datasets` tests: record frames, Cleveland-format text and archives.

The one piece of real data is `REAL_LAYOUT`, the real Cleveland file's first record, kept so the
parser is proven against the real float formatting. Every other line is built from it, and every
Synthea row is invented.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bko.datasets.fetch import snapshot_dir
from bko.datasets.sources import Source
from bko.datasets.uci import COLUMNS, MEMBER


def records(spec: dict[tuple[int, int, int], int]) -> pd.DataFrame:
    """Return `n` records for each `(age, sex, target): n`, indexed by `record_id` from 1."""
    rows = [cell for cell, n in spec.items() for _ in range(n)]
    frame = pd.DataFrame(rows, columns=["age", "sex", "target"], dtype="int64")
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame


FIXED_NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def pin(data: bytes, name: str = "uci-heart-disease") -> Source:
    """Return a `Source` whose size and sha256 are those of `data`."""
    return Source(
        name=name,
        url=f"https://example.invalid/{name}",
        filename=f"{name}.bin",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def plant(data_dir: Path, source: Source, data: bytes) -> Path:
    """Write `data` as `source`'s snapshot without fetching, and return the file."""
    path = snapshot_dir(source, data_dir) / source.filename
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


class FailingBody(io.BytesIO):
    """A response body that serves its first read, then raises `error` on the next."""

    def __init__(self, body: bytes, error: BaseException) -> None:
        super().__init__(body)
        self.error = error
        self.served = False

    def read(self, size: int | None = -1, /) -> bytes:
        """Return one chunk the first time, then raise."""
        if self.served:
            raise self.error
        self.served = True
        return super().read(size)


class FakeOpener:
    """Serve one body for any URL, or raise; record every call.

    `error` is raised on open, `mid_stream` after the first chunk, and `on_open` runs before the
    body is served.
    """

    def __init__(
        self,
        body: bytes = b"",
        error: BaseException | None = None,
        *,
        mid_stream: BaseException | None = None,
        on_open: Callable[[], object] | None = None,
    ) -> None:
        self.body = body
        self.error = error
        self.mid_stream = mid_stream
        self.on_open = on_open
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        if self.on_open is not None:
            self.on_open()
        if self.mid_stream is not None:
            return FailingBody(self.body, self.mid_stream)
        return io.BytesIO(self.body)


REAL_LAYOUT = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0"
"""The first line of the real `processed.cleveland.data`: floats, with `num` a bare integer."""


def cleveland_line(*, age: int = 63, sex: int = 1, num: int = 0, **fields: str) -> str:
    """Return a Cleveland line in `REAL_LAYOUT`'s format with the given fields replaced."""
    values: dict[str, str] = dict(zip(COLUMNS, REAL_LAYOUT.split(","), strict=True))
    values |= {"age": f"{age}.0", "sex": f"{sex}.0", "num": str(num), **fields}
    return ",".join(values[name] for name in COLUMNS)


def with_field(line: str, field: str, value: str) -> str:
    """Return the Cleveland `line` with `field` set to `value`."""
    values = dict(zip(COLUMNS, line.split(","), strict=True)) | {field: value}
    return ",".join(values[name] for name in COLUMNS)


def archive(text: str, member: str = MEMBER) -> bytes:
    """Return a zip holding `text` as `member`, with a fixed timestamp so its bytes repeat."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(zipfile.ZipInfo(member, date_time=(2023, 5, 22, 0, 0, 0)), text)
    return buffer.getvalue()
