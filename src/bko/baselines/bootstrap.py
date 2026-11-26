"""The label bootstrap: whole training records of the stratum's label drawn into the cell.

Each added row is one training record of the stratum's label whose age is in the stratum's band,
drawn uniformly with replacement, with its sex set to the cell's; a label stratum draws any record
of the label unchanged. Every other field stays as the record holds it, so every within-record
association survives. ADR `research/baseline-augmentation-design` records the choice.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.records import Baseline, cell_rows, derived_seed, joined, unit_weights
from bko.baselines.request import Request
from bko.datasets.cells import LABEL, cell_keys


logger = logging.getLogger(__name__)


def label_bootstrap(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each stratum's quota as records of its label group drawn with replacement.

    Raises:
        BaselineError: If a stratum has no training record of its label in its band.
    """
    frame = request.frame
    cells = cell_keys(train, edges=frame.edges)
    parts: list[pd.DataFrame] = []
    pools: dict[str, int] = {}
    for stratum, rows in request.quota.items():
        name = "/".join(map(str, stratum))
        keys = dict(zip(request.strata.columns, stratum, strict=True))
        held = train.loc[_pool(cells, keys), list(frame.columns)]
        if held.empty:
            msg = (
                f"label-bootstrap has no training record to draw for {name}: "
                f"{(cells[LABEL] == keys[LABEL]).sum()} of the label, "
                f"{_band_count(cells, keys)} in the band"
            )
            raise BaselineError(msg)
        pools[name] = len(held)
        logger.info("label-bootstrap stratum=%s pool=%d rows=%d", name, len(held), rows)
        rng = np.random.default_rng(derived_seed(request.seed, "label-bootstrap", *stratum))
        drawn = held.iloc[rng.integers(len(held), size=rows)].reset_index(drop=True)
        sex = str(keys["sex"]) if "sex" in keys else None
        parts.append(cell_rows(drawn, frame, sex=sex))
    added = joined(parts, frame)
    return Baseline(added=added, weights=unit_weights(train), params={"pool": pools})


def _pool(cells: pd.DataFrame, keys: dict[str, str | int]) -> pd.Series:
    """Which training records the stratum draws from: its label, and its band for a cell."""
    inside = cells[LABEL] == keys[LABEL]
    if "age_band" in keys:
        inside &= cells["age_band"] == keys["age_band"]
    return inside


def _band_count(cells: pd.DataFrame, keys: dict[str, str | int]) -> str:
    """The training records in the stratum's band, or `n/a` for a label stratum."""
    if "age_band" not in keys:
        return "n/a"
    return str((cells["age_band"] == keys["age_band"]).sum())
