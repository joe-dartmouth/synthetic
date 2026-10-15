"""Offline inputs for the `bko.datasets` tests: record frames, Cleveland-format text and archives.

The one piece of real data is `REAL_LAYOUT`, the real Cleveland file's first record, kept so the
parser is proven against the real float formatting. Every other line is built from it, and every
Synthea row is invented.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bko.datasets.fetch import snapshot_dir
from bko.datasets.sources import Source


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


class FakeOpener:
    """Serve one body for any URL, or raise; record every call."""

    def __init__(self, body: bytes = b"", error: BaseException | None = None) -> None:
        self.body = body
        self.error = error
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.body)
