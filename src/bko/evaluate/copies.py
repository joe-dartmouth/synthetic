"""Added rows that copy a test record, and the test split without the records they copy.

A generator may reproduce a public record word for word. A copy of a test record that joins
training would be scored by a classifier that trained on it, so the harness counts copies and
scores every arm on the test split without them (ADR `research/e1-evaluation-protocol`).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME


def _records(rows: pd.DataFrame, frame: FrameSpec) -> list[tuple[object, ...]]:
    return list(rows.loc[:, list(frame.columns)].itertuples(index=False, name=None))


def copies(rows: pd.DataFrame, test: pd.DataFrame, *, frame: FrameSpec = UCI_FRAME) -> int:
    """How many of `rows` equal some test record in every one of `frame`'s columns."""
    held = set(_records(test, frame))
    return sum(record in held for record in _records(rows, frame))


def decontaminated(
    test: pd.DataFrame, added: Iterable[pd.DataFrame], *, frame: FrameSpec = UCI_FRAME
) -> pd.DataFrame:
    """`test` without each record some frame of `added` copies in every one of `frame`'s columns."""
    seen = {record for rows in added for record in _records(rows, frame)}
    return test.loc[[record not in seen for record in _records(test, frame)]]
