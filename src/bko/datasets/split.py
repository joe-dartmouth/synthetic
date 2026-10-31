"""The train/test split: one record in five of every cell held out, chosen by hash.

Every stage before Evaluate reads `Split.train`, and only `bko.evaluate` reads `Split.test`, a
convention ADR `research/uci-evaluation-frame` records and no code enforces. Stratifying on the full
cell keeps each cell's training count a fixed function of its size, so the underrepresented cells
are the same for every seed. Records are ranked by `sha256(f"{seed}:{record_id}")`, which no
library upgrade can change. `subsampled` keeps a share of each cell's training rows for a
scarcity run, ranked the same way under its own tag so a smaller share nests in a larger one.
`uci_split` fetches UCI Heart Disease through `ensure`, loads and
splits it, and logs the split; `split` never fetches.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.cells import CELL_COLUMNS, cell_keys
from bko.datasets.errors import ContractError
from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import Source
from bko.datasets.uci import UCI_FRAME, load_uci


logger = logging.getLogger("bko.datasets")


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


def split(frame: pd.DataFrame, *, seed: int = SPLIT_SEED, edges: Sequence[int]) -> Split:
    """Hold out `held_out_count` records of every cell, those ranked lowest by hash.

    Args:
        frame: Records with a unique integer index and the columns `cell_keys` reads.
        seed: The ranking seed.
        edges: The age band starts of the cells; pass the ones `cell_counts` will count with.

    Returns:
        The partition, each part a frame of its own that can be changed without touching `frame`.

    Raises:
        ValueError: If the index has a duplicate.
    """
    if not frame.index.is_unique:
        msg = "split needs a unique record index"
        raise ValueError(msg)
    held_out: list[int] = []
    for records in cell_keys(frame, edges=edges).groupby(list(CELL_COLUMNS)).groups.values():
        ranked = sorted(records, key=lambda record: _rank(seed, record))
        held_out.extend(ranked[: held_out_count(len(ranked))])
    is_test = frame.index.isin(held_out)
    return Split(train=frame.loc[~is_test].copy(), test=frame.loc[is_test].copy(), seed=seed)


def kept_count(cell_size: int, fraction: Fraction) -> int:
    """How many of a cell's training rows a subsample keeps: `fraction` of them, rounded half up.

    Exact on the fraction, never below 2 so a cell keeps a spread, and never above the cell.
    """
    return min(cell_size, max(2, math.floor(fraction * cell_size + Fraction(1, 2))))


def subsampled(
    train: pd.DataFrame, fraction: Fraction, *, seed: int, edges: Sequence[int]
) -> pd.DataFrame:
    """`kept_count` of each cell's rows of `train`, those ranked lowest by hash.

    Records are ranked by `sha256(f"{seed}:train-fraction:{record_id}")`, so one seed's
    subsamples nest: every record a smaller fraction keeps, a larger one keeps too. A fraction of
    1 keeps every row.

    Raises:
        ValueError: If `fraction` is not in (0, 1].
    """
    if not 0 < fraction <= 1:
        msg = f"training fraction {fraction} must lie in (0, 1]"
        raise ValueError(msg)
    kept: list[int] = []
    for records in cell_keys(train, edges=edges).groupby(list(CELL_COLUMNS)).groups.values():
        ranked = sorted(records, key=lambda record: _rank(seed, f"train-fraction:{record}"))
        kept.extend(ranked[: kept_count(len(ranked), fraction)])
    return train.loc[train.index.isin(kept)].copy()


def ranges(train: pd.DataFrame, frame: FrameSpec) -> dict[str, tuple[float, float]]:
    """Each of `frame`'s numeric fields' minimum and maximum over the training split.

    Args:
        train: The training split.
        frame: The frame whose numeric fields are read.

    Returns:
        Each numeric field's minimum and maximum, in the frame's order.

    Raises:
        ContractError: If `train` holds no record, or a numeric field's minimum or maximum is
            not finite (an all-missing column, or an infinite value).
    """
    if train.empty:
        msg = "the training split holds no record"
        raise ContractError(msg)
    found = {name: (float(train[name].min()), float(train[name].max())) for name in frame.numeric}
    for name, bounds in found.items():
        if not all(math.isfinite(bound) for bound in bounds):
            msg = f"the training split's {name} has no finite range: {bounds[0]} to {bounds[1]}"
            raise ContractError(msg)
    return found


def _rank(seed: int, record: int | str) -> bytes:
    return hashlib.sha256(f"{seed}:{record}".encode("ascii")).digest()


def uci_split(
    data_dir: Path, *, source: Source, open_url: UrlOpener, seed: int = SPLIT_SEED
) -> Split:
    """Fetch UCI Heart Disease into `data_dir` if absent, load it, split it, and log the split.

    Raises:
        DatasetError: As `ensure` and `load_uci`.
    """
    ensure(source, data_dir, open_url=open_url)
    parts = split(load_uci(data_dir, source=source), seed=seed, edges=UCI_FRAME.edges)
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    return parts
