"""Each holdout's training rows: its split sized by the training fraction, after the removal.

Under a synthetic share the training split stays whole, and the rows the share replaces are named
beside it.

The harness (`bko.evaluate.run`) and the rehearsal (`bko.evaluate.rehearse`) both prepare a
holdout here, so their training rows agree.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256

import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.ablation import thinned
from bko.datasets.cells import SEXES, age_band_labels, cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split, dropped, subsampled
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
        dropped: The training rows of `ablated` the experiment's synthetic share replaces, which
            `ablated.train` still holds: every arm fits on them, and no classifier trains on
            them but `real`'s. Empty with no share.
        kept: The real rows a classifier trains on, `ablated.train` less `dropped`, computed once
            per holdout: `ablated.train` itself, uncopied, when no row is dropped. A reader that
            trains a classifier on a holdout's real rows reads them here, so a share's dropped
            rows stay out.
    """

    sized: Split
    ablated: Split
    removed: pd.DataFrame
    dropped: pd.DataFrame
    kept: pd.DataFrame


def prepared(parts: Split, experiment: Experiment, *, edges: Sequence[int]) -> Prepared:
    """`parts` sized by `experiment.training_fraction`, then thinned by `experiment.removal`.

    Both draws take the split's seed (`bko.datasets.split.subsampled`,
    `bko.datasets.ablation.thinned`), and the test rows are `parts`' own. A fraction below 1 logs
    one `subsample` line (the holdout, the fraction, and the training rows before and kept), and a
    removal one `removal` line. A synthetic share names its dropped rows at the split's seed
    (`bko.datasets.split.dropped`) and logs one `share` line: the holdout, the share as
    `<numerator>/<denominator>`, the rows kept and dropped, and the sha256 of the dropped ids,
    sorted and joined by commas.

    Raises:
        DatasetError: As `thinned`, if the removal's cell holds no training row.
    """
    sized = _sized(parts, experiment.training_fraction, edges=edges)
    ablated, removed = _ablated(sized, experiment.removal, edges=edges)
    lost = _dropped(ablated, experiment.synthetic_share, edges)
    train = ablated.train
    kept = train if lost.empty else train.drop(index=lost.index)
    return Prepared(sized, ablated, removed, lost, kept)


def share_text(share: Fraction) -> str:
    """A synthetic share as the logs spell it: `<numerator>/<denominator>`, `1/1` for 1."""
    return f"{share.numerator}/{share.denominator}"


def _dropped(parts: Split, share: Fraction | None, edges: Sequence[int]) -> pd.DataFrame:
    train = parts.train
    if share is None:
        return train.iloc[:0]
    ids = dropped(train, share, seed=parts.seed, edges=edges)
    digest = sha256(",".join(str(record) for record in sorted(ids)).encode("ascii")).hexdigest()
    logger.info(
        "share holdout=%d share=%s kept=%d dropped=%d dropped_sha256=%s",
        parts.seed,
        share_text(share),
        len(train) - len(ids),
        len(ids),
        digest,
    )
    return train.loc[ids]


def _sized(parts: Split, fraction: Fraction, *, edges: Sequence[int]) -> Split:
    if fraction == 1:
        return parts
    sized = replace(parts, train=subsampled(parts.train, fraction, seed=parts.seed, edges=edges))
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
    return replace(parts, train=kept), removed


def held_out_ratio(parts: Split, *, edges: Sequence[int]) -> float:
    """Nadeau and Bengio's n2/n1: the largest ratio of held-out to training records in a cell.

    Taking the largest keeps every scope's interval at least as wide as its own cells need. The
    harness reads it from the sized split (`Prepared.sized`): after the training fraction, which
    shrinks every training cell and so widens every interval, and before any removal, so one
    thinned cell does not set it. A fixed split's (`Split.fixed`) is 0: every holdout scores
    the same test records after the same training records, so its contrasts vary only by the
    arms' draws, independent across holdouts, and the plain t interval applies (ADR
    `research/tableshift-comparison`, Decision 7).
    """
    if parts.fixed:
        return 0.0
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


def require_cells(experiment: Experiment, frame: FrameSpec) -> None:
    """Refuse a removal's cell or a flagged cell that is not one of `frame`'s cells.

    Raises:
        EvaluationError: If either cell's band, sex label or label is not `frame`'s.
    """
    cells = {
        (band, sex, label)
        for band in age_band_labels(frame.edges)
        for sex in SEXES.values()
        for label in frame.label_codes
    }
    removal = experiment.removal
    if removal is not None and removal.cell not in cells:
        msg = f"the removal's cell {removal.cell} is not a cell of the {frame.name} frame"
        raise EvaluationError(msg)
    flagged = experiment.flagged_cell
    if flagged is not None and flagged not in cells:
        msg = f"the flagged cell {flagged} is not a cell of the {frame.name} frame"
        raise EvaluationError(msg)
