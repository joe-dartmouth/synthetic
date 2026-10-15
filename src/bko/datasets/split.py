"""The train/test split: one record in five of every cell held out, chosen by hash.

Every stage before Evaluate reads `Split.train`, and only `bko.evaluate` reads `Split.test` (ADR
`research/uci-evaluation-frame`). Stratifying on the full cell keeps each cell's training count a
fixed function of its size, so the underrepresented cells are the same for every seed. Records are
ranked by `sha256(f"{seed}:{record_id}")`, which no library upgrade can change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import pandas as pd

from bko.datasets.cells import CELL_COLUMNS, cell_keys


SPLIT_SEED: Final = 0
"""The seed of the split every stage uses unless an experiment's configuration fixes another."""

TEST_SHARE_DENOMINATOR: Final = 5
"""One record in this many of each cell is held out, rounded half up."""


@dataclass(frozen=True, slots=True, eq=False)
class Split:
    """A disjoint train/test partition of one frame.

    Attributes:
        train: The rows every stage before Evaluate reads.
        test: The held-out rows; only `bko.evaluate` reads them.
        seed: The seed that ranked the records.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    seed: int

    @property
    def fingerprint(self) -> str:
        """The sha256 of the sorted test `record_id`s, joined by commas."""
        ids = ",".join(str(record) for record in sorted(self.test.index))
        return hashlib.sha256(ids.encode("ascii")).hexdigest()


def held_out_count(cell_size: int) -> int:
    """Return how many of a cell's records are held out: one fifth of them, rounded half up."""
    return (2 * cell_size + TEST_SHARE_DENOMINATOR) // (2 * TEST_SHARE_DENOMINATOR)


def split(frame: pd.DataFrame, *, seed: int = SPLIT_SEED) -> Split:
    """Hold out `held_out_count` records of every cell, those ranked lowest by hash.

    Args:
        frame: Records with a unique integer index and the columns `cell_keys` reads.
        seed: The ranking seed.

    Returns:
        The partition.

    Raises:
        ValueError: If the index has a duplicate.
    """
    if not frame.index.is_unique:
        msg = "split needs a unique record index"
        raise ValueError(msg)
    held_out: list[int] = []
    for records in cell_keys(frame).groupby(list(CELL_COLUMNS)).groups.values():
        ranked = sorted(records, key=lambda record: _rank(seed, record))
        held_out.extend(ranked[: held_out_count(len(ranked))])
    is_test = frame.index.isin(held_out)
    return Split(train=frame.loc[~is_test], test=frame.loc[is_test], seed=seed)


def _rank(seed: int, record: int) -> bytes:
    return hashlib.sha256(f"{seed}:{record}".encode("ascii")).digest()
