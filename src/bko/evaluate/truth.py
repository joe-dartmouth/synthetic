"""`python -m bko.evaluate truth RUN --rows PATH...`: each fill against the records it replaces.

A run with a removal thins one cell of each holdout's training split before any arm runs
(`bko.evaluate.prepare.prepared`); the records it takes are that cell's truth, which no arm saw.
For each holdout, `truth` re-derives them, keeps each arm's gap fill rows in the removed cell, and
scores those rows against them with SDMetrics' quality report (`bko.evaluate.fidelity`), the
report a replica's fidelity is scored by (`bko.evaluate.run_scores`): the column shapes and
column pair trends scores, and each column's shape. `real`, whose rows are the removed records,
is left out. It reads the run's `rows` stores, one per slice, bound by the privacy reading's
checks (`bko.evaluate.privacy_stores`), makes no API call, and judges no criterion. ADR
`research/removed-truth-reading` records the reading.
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
from bko.datasets.frames import FrameSpec
from bko.datasets.run_folders import runs_dir_argument
from bko.evaluate.attack_stats import scopes
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import REAL_ARM, Experiment
from bko.evaluate.fidelity import ReportScores, quality_report, report_scores
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.privacy_stores import Store, checked, read, read_fills


logger = logging.getLogger("bko.evaluate")

TRUTH_COLUMNS: Final = ("holdout", "arm", "rows", "measure", "column", "value")
"""The CSV's columns: one row per holdout, arm and measure, each column's shape on its own row."""

Row = tuple[int, str, int, str, str, float]
"""One CSV row, in `TRUTH_COLUMNS`' order."""

MIN_ROWS: Final = 2
"""The fewest rows either side must hold to be scored.

It mirrors the replica rule (`bko.evaluate.run_scores.MIN_LABELS`): a replica of fewer rows is
not scored, so neither is a fill or a removal of fewer.
"""


def truth_scores(added: pd.DataFrame, removed: pd.DataFrame, *, frame: FrameSpec) -> ReportScores:
    """SDMetrics' column shapes, pair trends and each column's shape of `added` against `removed`.

    Every score is NaN when either side holds fewer than `MIN_ROWS` rows; a score the report
    lacks reads NaN (`report_scores`).
    """
    if len(added) < MIN_ROWS or len(removed) < MIN_ROWS:
        return ReportScores(math.nan, math.nan, dict.fromkeys(frame.columns, math.nan))
    report = quality_report(removed, added, frame=frame)
    return report_scores(report, columns=frame.columns)


def truth(
    run: str,
    stores: Sequence[Path],
    *,
    experiment: Experiment,
    data: Data,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> str:
    """Read each arm's fill of `run` against its removed records from `stores`; return the CSV.

    It logs `truth run=%s stores=%d holdouts=%d cell=%s fraction=%s min_rows=%d root=%s`,
    `read`'s store lines, then per holdout `prepared`'s `removal` line,
    `truth holdout=%d removed=%d`, one `truth holdout=%d arm=%s rows=%d column_shapes=%.4f
    pair_trends=%.4f` line per arm, and `truth holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f`;
    then `truth lines=%d sha256=%s seconds=%.1f`, `lines` the CSV's rows. An arm line whose side
    holds fewer than `MIN_ROWS` rows ends ` short=added`, ` short=removed` or
    ` short=added,removed` and logs at INFO; one with a NaN score and enough rows on both sides
    logs at WARNING.

    Raises:
        EvaluationError: If the experiment removes no cell.
        IdentityError: As `read`, `checked` and `read_fills`, naming the store.
    """
    begun = time.perf_counter()
    removal = experiment.removal
    if removal is None:
        msg = f"run {run!r} removes no cell, so it has no removed records to read a fill against"
        raise EvaluationError(msg)
    total = len(experiment.holdout_seeds)
    logger.info(
        "truth run=%s stores=%d holdouts=%d cell=%s fraction=%s min_rows=%d root=%s",
        run,
        len(stores),
        total,
        "/".join(str(value) for value in removal.cell),
        removal.fraction,
        MIN_ROWS,
        root,
    )
    held = read(run, stores, data, experiment=experiment, runs=runs, root=root, verb="truth")
    arms = [arm for arm in experiment.arms if arm != REAL_ARM]
    rows: list[Row] = []
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started = time.perf_counter()
        rows += _holdout(
            seed, held[seed], data, experiment=experiment, cell=removal.cell, arms=arms
        )
        now = time.perf_counter()
        logger.info(
            "truth holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f",
            seed,
            done,
            total,
            now - started,
            now - begun,
        )
    text = pd.DataFrame(rows, columns=list(TRUTH_COLUMNS)).to_csv(index=False, lineterminator="\n")
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info(
        "truth lines=%d sha256=%s seconds=%.1f", len(rows), sha, time.perf_counter() - begun
    )
    return text


def _holdout(
    seed: int,
    store: Store,
    data: Data,
    *,
    experiment: Experiment,
    cell: Stratum,
    arms: Sequence[str],
) -> list[Row]:
    """Holdout `seed`'s rows: each of `arms`' rows in the removed cell against the removed records.

    Raises:
        IdentityError: As `checked` and `read_fills`.
    """
    frame = data.frame
    removed = checked(seed, store, data, experiment).removed
    logger.info("truth holdout=%d removed=%d", seed, len(removed))
    rows: list[Row] = []
    for arm, added in read_fills(seed, store, arms, frame).items():
        inside = added.loc[scopes(added, cell, edges=frame.edges)["cell"]]
        found = truth_scores(inside, removed, frame=frame)
        count = len(inside)
        short = [
            side for side, held in (("added", count), ("removed", len(removed))) if held < MIN_ROWS
        ]
        unscored = math.isnan(found.shapes) or math.isnan(found.pair_trends)
        level = logging.WARNING if unscored and not short else logging.INFO
        logger.log(
            level,
            "truth holdout=%d arm=%s rows=%d column_shapes=%.4f pair_trends=%.4f%s",
            seed,
            arm,
            count,
            found.shapes,
            found.pair_trends,
            f" short={','.join(short)}" if short else "",
        )
        rows += [
            (seed, arm, count, "column-shapes", "", found.shapes),
            (seed, arm, count, "column-pair-trends", "", found.pair_trends),
            *((seed, arm, count, "column-shape", c, v) for c, v in found.columns.items()),
        ]
    return rows


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `truth` subcommand to `commands`."""
    reading = commands.add_parser(
        "truth", help="read each fill of a run with a removal against the records it removed"
    )
    reading.add_argument("run", help="the named run, one with a removal")
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
