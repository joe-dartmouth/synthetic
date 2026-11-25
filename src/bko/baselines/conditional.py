"""Conditional sequential CART: some columns of given rows redrawn from trees fit on the others.

`conditional_cart` keeps each given row's `given` columns and redraws its `visit` columns in
order, each from a tree fit on the training rows' `given` columns and the columns visited before
it, with `bko.baselines.classical.cart`'s settings, predictors and leaf draw: a visited value is
the value of a training row drawn uniformly from the rows in the new row's leaf. It also returns
that training row's record id, the training frame's index, for each value. ADR
`research/free-field-synthesis` records the draw `bko.evaluate.coherent` makes with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from bko.baselines.classical import fit_tree, leaf_donors, tree_predictors
from bko.datasets.frames import FrameSpec


def conditional_cart(
    train: pd.DataFrame,
    drawn: pd.DataFrame,
    *,
    given: Sequence[str],
    visit: Sequence[str],
    frame: FrameSpec,
    rng: np.random.Generator,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`drawn` with its `visit` columns redrawn given its `given` columns, and each value's donor.

    One tree per visited column is fit on `train`, predicting it from `given` and the visited
    columns before it, at `seed` as `cart` seeds its trees. Each tree then reads `drawn`'s rows as
    redrawn so far, and each row takes the value of a training row drawn by `rng` uniformly from
    its leaf. With no row drawn, no tree is fit and nothing is drawn.

    Args:
        train: The training rows the trees fit on and the values come from, indexed by record id.
        drawn: The rows to redraw, at `frame`'s columns.
        given: The columns kept, the predictors of every tree.
        visit: The columns redrawn, in the order they are visited.
        frame: The frame whose codes decide each tree's kind and its predictors' coding.
        rng: The generator every leaf draw reads.
        seed: The seed each tree's own `random_state` derives from.

    Returns:
        `drawn` with its `visit` columns replaced and `frame`'s dtypes, on `drawn`'s index; and
        one column per visited field, on that index, holding the training record id its value
        was copied from.

    Raises:
        ValueError: If a `given` or `visit` entry is not a column of `frame`, if `given` is
            empty, or if `given` and `visit` share a column.
    """
    columns = [*given, *visit]
    for name in columns:
        if name not in frame.columns:
            msg = f"column {name!r} is not a column of frame {frame.name!r}"
            raise ValueError(msg)
    if not given or len(set(columns)) != len(columns):
        msg = f"given {list(given)} and visit {list(visit)} must be distinct columns of the frame,"
        msg += " given at least one"
        raise ValueError(msg)
    out = drawn.copy()
    picks = {name: np.empty(0, dtype=np.int64) for name in visit}
    if len(drawn):
        rows = train.loc[:, columns]
        ids = train.index.to_numpy(dtype=np.int64)
        for at in range(len(given), len(columns)):
            _, tree = fit_tree(rows, columns, at, frame, seed=seed)
            trained = tree.apply(tree_predictors(rows, columns[:at], frame))
            new = tree.apply(tree_predictors(out, columns[:at], frame))
            picked = leaf_donors(new, trained, rng)
            column = columns[at]
            out[column] = rows[column].to_numpy()[picked]
            picks[column] = ids[picked]
    donors = pd.DataFrame(picks, index=drawn.index, columns=list(visit), dtype=np.int64)
    return out.astype(dict(frame.dtypes)), donors
