"""Each holdout's training rows: its split after the experiment's removal, and what it reads.

The harness (`bko.evaluate.run`) and the rehearsal (`bko.evaluate.rehearse`) both prepare a
holdout here, so their training rows agree.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from hashlib import sha256

import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.ablation import thinned
from bko.datasets.cells import SEXES, age_band_labels, cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Removal


logger = logging.getLogger(__name__)


def prepared(
    parts: Split, removal: Removal | None, *, edges: Sequence[int]
) -> tuple[Split, pd.DataFrame]:
    """`parts` with `removal`'s rows gone from its training rows, and the rows removed.

    The draw takes the split's seed (`bko.datasets.ablation.thinned`), and the test rows are
    `parts`' own. With no removal, `parts` itself and no row. A removal logs one `removal` line:
    the holdout, the cell, the fraction, and the training rows before, removed and kept.

    Raises:
        DatasetError: As `thinned`, if the removal's cell holds no training row.
    """
    if removal is None:
        return parts, parts.train.iloc[:0]
    kept, removed = thinned(
        parts.train, cell=removal.cell, edges=edges, fraction=removal.fraction, seed=parts.seed
    )
    logger.info(
        "removal holdout=%d cell=%s fraction=%s train=%d removed=%d kept=%d",
        parts.seed,
        "/".join(map(str, removal.cell)),
        removal.fraction,
        len(parts.train),
        len(removed),
        len(kept),
    )
    return Split(kept, parts.test, parts.seed), removed


def held_out_ratio(parts: Split, *, edges: Sequence[int]) -> float:
    """Nadeau and Bengio's n2/n1: the largest ratio of held-out to training records in a cell.

    Taking the largest keeps every scope's interval at least as wide as its own cells need. The
    harness reads it from the split before any removal, so a thinned cell does not set it.
    """
    kept = cell_counts(parts.train, edges=edges)["count"].tolist()
    held = cell_counts(parts.test, edges=edges)["count"].tolist()
    return max(out / into for out, into in zip(held, kept, strict=True) if out)


def fingerprint(cells: Sequence[Stratum]) -> float:
    """`cells` as a number: the first 12 hex digits of the sha256 of their sorted list's `repr`.

    The value is below 2**48, so a float holds it exactly.
    """
    return float(int(sha256(repr(sorted(cells)).encode()).hexdigest()[:12], 16))


def require_cell(removal: Removal | None, frame: FrameSpec) -> None:
    """Refuse a removal whose cell is not one of `frame`'s cells.

    Raises:
        EvaluationError: If the cell's band, sex label or label is not `frame`'s.
    """
    if removal is None:
        return
    cells = {
        (band, sex, label)
        for band in age_band_labels(frame.edges)
        for sex in SEXES.values()
        for label in frame.label_codes
    }
    if removal.cell not in cells:
        msg = f"the removal's cell {removal.cell} is not a cell of the {frame.name} frame"
        raise EvaluationError(msg)
