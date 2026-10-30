"""The CLLM arms' shown records: each call shows training records drawn by a seeded generator.

CLLM (Seedat et al. 2024) prompts an LLM with real records and curates what it writes. Here each
call of a CLLM arm shows `SHOW` training records: the call's cell's own first, then the rest of its
sex-and-label group, then the rest of its label group, each group in a seeded random order and
none twice in one call. The generator is seeded once per attempt and advanced in planning order,
so a replay and every in-flight count show the same records.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.records import derived_seed
from bko.datasets.cells import cell_keys
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell
from bko.generate.prompt import Shown


SHOW: Final = 20
"""The training records each call shows: CLLM's code shows 20 per call, and `per_call` is 20."""


class Shower:
    """The records each call of one attempt shows, drawn in planning order."""

    def __init__(self, train: pd.DataFrame, *, frame: FrameSpec, seed: int) -> None:
        """Hold `train` in `frame`'s columns and seed the draw from the attempt's `seed`.

        Args:
            train: The holdout's training rows.
            frame: The frame whose columns are shown and whose band edges name the cells.
            seed: The attempt's seed; the draw takes `derived_seed(seed, "cllm")`.
        """
        rows = train.loc[:, list(frame.columns)].reset_index(drop=True)
        self._records = [
            {
                name: float(record[name]) if name in frame.fractional else int(record[name])
                for name in frame.columns
            }
            for record in rows.to_dict("records")
        ]
        keys = cell_keys(rows, edges=frame.edges)
        self._cells = list(keys.itertuples(index=False, name=None))
        self._rng = np.random.default_rng(derived_seed(seed, "cllm"))

    def take(self, cell: Cell) -> Shown:
        """The next call's shown records for `cell`: up to `SHOW`, its own cell's first."""
        own = [at for at, key in enumerate(self._cells) if key == tuple(cell)]
        group = [
            at
            for at, key in enumerate(self._cells)
            if key[1:] == tuple(cell[1:]) and key != tuple(cell)
        ]
        label = [
            at for at, key in enumerate(self._cells) if key[2] == cell[2] and key[1] != cell[1]
        ]
        order = [at for tier in (own, group, label) for at in self._rng.permutation(tier).tolist()]
        return tuple(self._records[at] for at in order[:SHOW])
