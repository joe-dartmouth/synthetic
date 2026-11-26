"""`python -m bko.evaluate representativeness RUN --rows PATH...`: each fill against fused shares.

The reading measures how far each fill sits from the fused shares of the cells it fills.

For each holdout, arm and flagged cell, the reading takes the arm's gap fill rows in the cell and
compares each mapped concept's share among them with the concept's fused share in the cell
(`bko.gaps.quota.fused_shares`, the shares the direct sampler draws at); the gap is the largest
absolute difference. Beside it, the cell's share of the arm's training set (the training split
plus the fill) is read against `real`'s, whose fill is the rows the removal took, and the arm's
recall on the flagged subgroups is read from the run's `scores.csv`. It reads the run's `rows`
stores, one per slice, bound by the privacy reading's checks (`bko.evaluate.privacy_stores`),
makes no API call, and judges no criterion. ADR `research/share-based-quota` records the reading.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.run_folders import runs_dir_argument
from bko.evaluate.attack_stats import scopes
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import REAL_ARM, Experiment, flagged
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.privacy_stores import Store, checked, read, read_fills
from bko.evaluate.rows import SCORES_CSV, folder_scores
from bko.evaluate.scores import Measure, Scope, Source
from bko.gaps.observe import cell_key, record_shows
from bko.gaps.quota import fused_shares
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.evaluate")

REPRESENTATIVENESS_COLUMNS: Final = (
    "holdout",
    "arm",
    "cell",
    "rows",
    "gap",
    "concept",
    "cell_share",
    "real_share",
    "recall",
)
"""The CSV's columns: one row per holdout, arm and flagged cell."""

Row = tuple[int, str, str, int, float, str, float, float, float]
"""One CSV row, in `REPRESENTATIVENESS_COLUMNS`' order."""


def share_gap(
    added: pd.DataFrame, fused: Mapping[str, float], rows: Sequence[FieldMapping], *, name: str
) -> tuple[float, str]:
    """The largest absolute gap between `added`'s concept shares and `fused`, and its concept.

    A concept's share among `added` is the fraction of its rows that show it under `rows`; a tie
    goes to the first concept in sorted order. NaN and an empty concept when `added` holds no row
    or `fused` names no concept.
    """
    if not len(added) or not fused:
        return math.nan, ""
    shows = record_shows(added, rows, name=name)
    gaps = {
        concept: abs(float(shows[concept].mean()) - share)
        for concept, share in sorted(fused.items())
    }
    concept = max(gaps, key=lambda each: gaps[each])
    return gaps[concept], concept


def representativeness(
    run: str,
    stores: Sequence[Path],
    *,
    experiment: Experiment,
    data: Data,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> str:
    """Read each arm's gap fill of `run` against its cells' fused shares; return the CSV.

    It logs `representativeness run=%s stores=%d holdouts=%d root=%s`, `read`'s store lines,
    then per holdout, the first time it reads a store, `representativeness scores folder=%s
    sha256=%s`, the sha256 of the store's folder's `scores.csv`, then one
    `representativeness holdout=%d arm=%s key=%s rows=%d gap=%.4f concept=%s cell_share=%.6f
    real_share=%.6f recall=%.4f` line per arm and flagged cell and
    `representativeness holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f`; then
    `representativeness lines=%d sha256=%s seconds=%.1f`, `lines` the CSV's rows. `recall` is
    the arm's regression recall on the flagged subgroups of its augmented training set, from the
    store's folder's `scores.csv` once its sha256 is the one the folder's `scores` line logs
    (`bko.evaluate.rows.folder_scores`); NaN where that file holds no such score.

    Raises:
        EvaluationError: If the experiment lacks `REAL_ARM`, whose training set the cell share
            is read beside, or a store's folder holds no `scores.csv`.
        IdentityError: As `read`, `checked` and `read_fills`, naming the store, or as
            `folder_scores`, if a folder's `scores.csv` is not the one its log names.
        ScoresFormatError: As `folder_scores`.
    """
    begun = time.perf_counter()
    if REAL_ARM not in experiment.arms:
        msg = f"run {run!r} lacks {REAL_ARM!r}, whose training set the cell share is read beside"
        raise EvaluationError(msg)
    total = len(experiment.holdout_seeds)
    logger.info(
        "representativeness run=%s stores=%d holdouts=%d root=%s", run, len(stores), total, root
    )
    held = read(
        run, stores, data, experiment=experiment, runs=runs, root=root, verb="representativeness"
    )
    recalls: dict[Path, dict[tuple[int, str], float]] = {}
    rows: list[Row] = []
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started = time.perf_counter()
        store = held[seed]
        if store.path not in recalls:
            recalls[store.path] = _recalls(store)
        rows += _holdout(seed, store, data, experiment=experiment, recall=recalls[store.path])
        now = time.perf_counter()
        logger.info(
            "representativeness holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f",
            seed,
            done,
            total,
            now - started,
            now - begun,
        )
    columns = list(REPRESENTATIVENESS_COLUMNS)
    text = pd.DataFrame(rows, columns=columns).to_csv(index=False, lineterminator="\n")
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info(
        "representativeness lines=%d sha256=%s seconds=%.1f",
        len(rows),
        sha,
        time.perf_counter() - begun,
    )
    return text


def _recalls(store: Store) -> dict[tuple[int, str], float]:
    """Each holdout and arm's regression recall on the flagged subgroups, from `store`'s folder.

    Raises:
        EvaluationError: If the folder holds no `scores.csv`.
        IdentityError, ScoresFormatError: As `folder_scores`.
    """
    folder = store.folder
    if not (folder.path / SCORES_CSV).is_file():
        msg = f"the store {store.path}'s folder {folder.path} holds no {SCORES_CSV}"
        raise EvaluationError(msg)
    scores = folder_scores(folder)
    logger.info("representativeness scores folder=%s sha256=%s", folder.path, folder.scores_sha256)
    return {
        (score.holdout, score.arm): score.value
        for score in scores
        if (score.source, score.measure, score.scope)
        == (Source.AUGMENTED, Measure.RECALL, Scope.FLAGGED)
    }


def _holdout(
    seed: int,
    store: Store,
    data: Data,
    *,
    experiment: Experiment,
    recall: Mapping[tuple[int, str], float],
) -> list[Row]:
    """Holdout `seed`'s rows: each arm's fill against each flagged cell's fused shares.

    Raises:
        IdentityError: As `checked` and `read_fills`.
    """
    frame = data.frame
    train = checked(seed, store, data, experiment).ablated.train
    cells: tuple[Stratum, ...] = flagged(train, experiment, edges=frame.edges)
    fused = fused_shares(train, cells, name=data.name, rows=data.field_rows, frame=frame)
    fills = read_fills(seed, store, experiment.arms, frame)
    rows: list[Row] = []
    for cell in cells:
        own = int(scopes(train, cell, edges=frame.edges)["cell"].sum())
        real = fills[REAL_ARM]
        real_inside = int(scopes(real, cell, edges=frame.edges)["cell"].sum())
        real_share = (own + real_inside) / (len(train) + len(real))
        for arm, added in fills.items():
            inside = added.loc[scopes(added, cell, edges=frame.edges)["cell"]]
            gap, concept = share_gap(inside, fused[cell], data.field_rows, name=data.name)
            share = (own + len(inside)) / (len(train) + len(added))
            found = recall.get((seed, arm), math.nan)
            logger.info(
                "representativeness holdout=%d arm=%s key=%s rows=%d gap=%.4f concept=%s"
                " cell_share=%.6f real_share=%.6f recall=%.4f",
                seed,
                arm,
                cell_key(cell),
                len(inside),
                gap,
                concept or "none",
                share,
                real_share,
                found,
            )
            rows.append(
                (seed, arm, cell_key(cell), len(inside), gap, concept, share, real_share, found)
            )
    return rows


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `representativeness` subcommand to `commands`."""
    reading = commands.add_parser(
        "representativeness",
        help="read each fill of a run against the fused shares of the cells it fills",
    )
    reading.add_argument("run", help="the named run, one holding `real`")
    reading.add_argument(
        "--rows",
        dest="stores",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="a store `rows` wrote, one per slice of the run; repeatable",
    )
    reading.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
    reading.add_argument(
        "--runs-dir",
        type=runs_dir_argument,
        default=None,
        help="the root the stores' folders sit under; default: runs/ beside the code",
    )
