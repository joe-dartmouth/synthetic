"""Offline inputs for the `bko.datasets` tests: record frames, Cleveland-format text and archives.

The one piece of real data is `REAL_LAYOUT`, the real Cleveland file's first record, kept so the
parser is proven against the real float formatting. Every other line is built from it, and every
Synthea row is invented.
"""

from __future__ import annotations

import pandas as pd


def records(spec: dict[tuple[int, int, int], int]) -> pd.DataFrame:
    """Return `n` records for each `(age, sex, target): n`, indexed by `record_id` from 1."""
    rows = [cell for cell, n in spec.items() for _ in range(n)]
    frame = pd.DataFrame(rows, columns=["age", "sex", "target"], dtype="int64")
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame
