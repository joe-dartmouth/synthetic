"""A holdout scored on a split shared with other runs: their copies leave it, and a row marks it.

Two runs whose arms copy different test records score them on different splits under E1's rule,
so `compare --beside` cannot pair those holdouts. Scoring each run again with the other runs'
added and replica rows as `exclude` drops every record any of them copies, so both runs score on
one split; the `shared-split` held-out row fingerprints that split's record ids, and `beside`
pairs a copied holdout when both files hold equal rows there (ADR
`research/shared-test-split-reading`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from hashlib import sha256

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.evaluate.copies import decontaminated
from bko.evaluate.scores import Measure


logger = logging.getLogger(__name__)


def split_fingerprint(kept: pd.DataFrame) -> str:
    """The first 12 hex digits of the sha256 of `kept`'s sorted integer record ids' `repr`."""
    return sha256(repr(sorted(int(i) for i in kept.index)).encode()).hexdigest()[:12]


def marker(kept: pd.DataFrame) -> dict[Measure, float]:
    """The `shared-split` held-out value of a split of `kept` records, read as an integer.

    It is below 2**48, so a float holds it exactly.
    """
    return {Measure.SHARED_SPLIT: float(int(split_fingerprint(kept), 16))}


def marked(
    scored: pd.DataFrame,
    exclude: Sequence[pd.DataFrame],
    *,
    records: int,
    copied: int,
    holdout: int,
    frame: FrameSpec,
) -> pd.DataFrame:
    """`scored` without each record a frame of `exclude` copies; `marker` reads the result.

    A record is copied when an `exclude` row equals it in every one of `frame`'s columns. The
    `shared` line logs the holdout's whole test split (`records`), the records its own arms copy
    (`copied`), the further records `exclude` drops (`foreign`) and the kept split's fingerprint.

    Args:
        scored: The test records the run's own rule scores on.
        exclude: Other runs' added and replica rows.
        records: The test split's records before any copy is dropped.
        copied: The test records the run's own arms copy.
        holdout: The holdout's seed.
        frame: The records' frame.

    Returns:
        The kept records.
    """
    kept = decontaminated(scored, exclude, frame=frame)
    logger.info(
        "shared holdout=%d records=%d copied=%d foreign=%d fingerprint=%s",
        holdout,
        records,
        copied,
        len(scored) - len(kept),
        split_fingerprint(kept),
    )
    return kept
