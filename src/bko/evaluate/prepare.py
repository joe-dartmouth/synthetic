"""Each holdout's training rows: its split sized by the training fraction, after the removal.

The harness (`bko.evaluate.run`) and the rehearsal (`bko.evaluate.rehearse`) both prepare a
holdout here, so their training rows agree.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.ablation import thinned
from bko.datasets.cells import SEXES, age_band_labels, cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split, subsampled
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment, Removal


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class Prepared:
    """One holdout's split at each step of its preparation.

    Attributes:
        sized: The split with the experiment's training fraction of each cell's training rows;
            the held-out ratio reads it.
        ablated: `sized` after the experiment's removal; every arm and score reads it.
        removed: The training rows the removal took.
    """

    sized: Split
    ablated: Split
    removed: pd.DataFrame


def prepared(parts: Split, experiment: Experiment, *, edges: Sequence[int]) -> Prepared:
    """`parts` sized by `experiment.training_fraction`, then thinned by `experiment.removal`.

    Both draws take the split's seed (`bko.datasets.split.subsampled`,
    `bko.datasets.ablation.thinned`), and the test rows are `parts`' own. A fraction below 1 logs
    one `subsample` line (the holdout, the fraction, and the training rows before and kept), and a
    removal one `removal` line.

    Raises:
        DatasetError: As `thinned`, if the removal's cell holds no training row.
    """
    sized = _sized(parts, experiment.training_fraction, edges=edges)
    ablated, removed = _ablated(sized, experiment.removal, edges=edges)
    return Prepared(sized, ablated, removed)


def _sized(parts: Split, fraction: Fraction, *, edges: Sequence[int]) -> Split:
    if fraction == 1:
        return parts
    sized = Split(
        subsampled(parts.train, fraction, seed=parts.seed, edges=edges), parts.test, parts.seed
    )
    logger.info(
        "subsample holdout=%d fraction=%s train=%d kept=%d ratio=%.6f",
        parts.seed,
        fraction,
        len(parts.train),
        len(sized.train),
        held_out_ratio(sized, edges=edges),
    )
    return sized


def _ablated(
    parts: Split, removal: Removal | None, *, edges: Sequence[int]
) -> tuple[Split, pd.DataFrame]:
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
    harness reads it from the sized split (`Prepared.sized`): after the training fraction, which
    shrinks every training cell and so widens every interval, and before any removal, so one
    thinned cell does not set it.
    """
    kept = cell_counts(parts.train, edges=edges)["count"].tolist()
    held = cell_counts(parts.test, edges=edges)["count"].tolist()
    return max(out / into for out, into in zip(held, kept, strict=True) if out)


def split_line(ablated: Split) -> str:
    """The harness's `split` line for a holdout's prepared split: seed, row counts, fingerprint."""
    rows = f"train={len(ablated.train):d} test={len(ablated.test):d}"
    return f"split seed={ablated.seed:d} {rows} fingerprint={ablated.fingerprint}"


def sorted_digest(items: Iterable[Stratum] | Iterable[int]) -> str:
    """The first 12 hex digits of the sha256 of `items`' sorted list's `repr`.

    Read as an integer, it is below 2**48, so a float holds it exactly.
    """
    return sha256(repr(sorted(items)).encode()).hexdigest()[:12]


def fingerprint(cells: Sequence[Stratum]) -> float:
    """`cells` as a number: `sorted_digest` of them, read as an integer."""
    return float(int(sorted_digest(cells), 16))


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
