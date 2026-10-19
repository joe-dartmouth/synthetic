"""Added rows that copy a test record, and the test split without the records they copy.

A generator may reproduce a public record word for word. A copy of a test record that joins
training would be scored by a classifier that trained on it, so the harness counts copies and
scores every arm on the test split without them (ADR `research/e1-evaluation-protocol`).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bko.datasets.uci import FRAME_COLUMNS


def _records(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    return list(frame.loc[:, list(FRAME_COLUMNS)].itertuples(index=False, name=None))


def copies(rows: pd.DataFrame, test: pd.DataFrame) -> int:
    """How many of `rows` equal some test record in every `FRAME_COLUMNS` value."""
    held = set(_records(test))
    return sum(record in held for record in _records(rows))


def decontaminated(test: pd.DataFrame, added: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """`test` without each record that a frame of `added` copies in every `FRAME_COLUMNS` value."""
    seen = {record for frame in added for record in _records(frame)}
    return test.loc[[record not in seen for record in _records(test)]]
