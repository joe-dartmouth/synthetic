"""A partial gap fill: the rows a generator kept when it reached its row limit short of a quota.

`bko.evaluate.run` retries a refused fill; on a gap fill's last attempt, a `ShortfallError` that
holds the method's baseline is scored as the rows it kept rather than refused (ADR
`research/e1-evaluation-protocol`). The kept rows are checked against the quota less the shortfall,
so a partial fill keeps every promise a full one does for the cells it fills.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping

import pandas as pd

from bko.baselines.errors import BaselineError, OutputContractError, ShortfallError
from bko.baselines.records import Baseline, check
from bko.baselines.request import Request, Stratum


logger = logging.getLogger(__name__)


def partial_fill(
    baseline: Baseline,
    short: Mapping[str, int],
    train: pd.DataFrame,
    request: Request,
    *,
    fill: tuple[int, str, int],
) -> Baseline:
    """`baseline`, the rows a generator kept `short` of `request`, checked against what it filled.

    `fill` is the holdout, the arm's name and the attempt. Logs one `shortfall` line per short
    cell; the caller logs the `arm` line, whose params name the shortfall under `short`.

    Raises:
        OutputContractError: If `short` names a cell the request does not ask for or more rows than
            it asks of one, or the kept rows break the contract for the cells they fill.
    """
    holdout, name, attempt = fill
    quota = {"/".join(map(str, cell)): (cell, rows) for cell, rows in request.quota.items()}
    try:
        filled = quota_filled(short, quota, name)
        check(baseline, train, dataclasses.replace(request, quota=filled))
    except OutputContractError:
        logger.error(  # ruff: ignore[error-instead-of-exception]
            "contract holdout=%d arm=%s attempt=%d", holdout, name, attempt
        )
        raise
    for key, rows in short.items():
        logger.warning(
            "shortfall holdout=%d arm=%s attempt=%d cell=%s short=%d quota=%d",
            holdout,
            name,
            attempt,
            key,
            rows,
            quota[key][1],
        )
    return baseline


def shortfall_kept(exc: BaselineError) -> tuple[Baseline, Mapping[str, int]] | None:
    """A shortfall's kept rows and missing rows, if `exc` is a shortfall that holds a baseline."""
    if isinstance(exc, ShortfallError) and exc.baseline is not None:
        return exc.baseline, exc.short
    return None


def quota_filled(
    short: Mapping[str, int], quota: Mapping[str, tuple[Stratum, int]], name: str
) -> dict[Stratum, int]:
    """Each cell's rows less `short`, cells left empty dropped.

    Raises:
        OutputContractError: If `short` names a cell `quota` lacks, or a count outside 1 to its
            cell's quota.
    """
    if not set(short) <= set(quota) or any(not 0 < n <= quota[k][1] for k, n in short.items()):
        msg = f"{name}'s shortfall {dict(short)} does not fit its quota"
        raise OutputContractError(msg)
    filled = {cell: rows - short.get(key, 0) for key, (cell, rows) in quota.items()}
    return {cell: rows for cell, rows in filled.items() if rows}
