"""The label bootstrap: whole training records of the stratum's label drawn into the cell.

Each added row is one training record of the stratum's label whose age is in the stratum's band,
drawn uniformly with replacement, with its sex set to the cell's; a label stratum draws any record
of the label unchanged. Every other field stays as the record holds it, so every within-record
association survives. ADR `research/baseline-augmentation-design` records the choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.records import Baseline, derived_seed, no_rows, number, unit_weights
from bko.baselines.request import Request, Stratum
from bko.datasets.cells import LABEL, SEXES, cell_keys


def label_bootstrap(train: pd.DataFrame, request: Request) -> Baseline:
    """Add each stratum's quota as records of its label group drawn with replacement.

    Raises:
        BaselineError: If a stratum has no training record of its label in its band.
    """
    frame = request.frame
    cells = cell_keys(train, edges=frame.edges)
    codes = {name: code for code, name in SEXES.items()}
    parts: list[pd.DataFrame] = []
    pools: dict[str, int] = {}
    for stratum, rows in request.quota.items():
        name = "/".join(map(str, stratum))
        held = train.loc[_pool(cells, stratum), list(frame.columns)]
        if held.empty:
            msg = f"label-bootstrap has no training record to draw for {name}"
            raise BaselineError(msg)
        pools[name] = len(held)
        rng = np.random.default_rng(derived_seed(request.seed, "label-bootstrap", *stratum))
        drawn = held.iloc[rng.integers(len(held), size=rows)].reset_index(drop=True)
        if len(stratum) > 1:
            drawn["sex"] = codes[str(stratum[1])]
        parts.append(drawn.astype(dict(frame.dtypes)))
    added = number(pd.concat(parts)) if parts else no_rows(frame)
    return Baseline(added=added, weights=unit_weights(train), params={"pool": pools})


def _pool(cells: pd.DataFrame, stratum: Stratum) -> pd.Series:
    """Which training records the stratum draws from: its label, and its band for a cell."""
    inside = cells[LABEL] == stratum[-1]
    if len(stratum) > 1:
        inside &= cells["age_band"] == stratum[0]
    return inside
