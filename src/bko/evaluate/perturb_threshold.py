"""`python -m bko.evaluate perturb-threshold`: the distribution check's delta from stored rows.

For each holdout of the named run, in seed order, it reads `bko-direct`'s gap fill U from the
run's `rows` stores (`bko.evaluate.membership.read` binds each store to its folder, and the
holdout's `split` line must be the one the data prepares) and the label-1 rows R of the holdout's
prepared training split, and reads SDMetrics' column shapes and pair trends of U against R over
the unmapped fields (`bko.evaluate.perturb.quality`). delta is the 95th percentile, by linear
interpolation, of the absolute step in shapes between consecutive holdouts; a shape or delta that
is not finite is refused. It writes one CSV row per holdout (`holdout,shapes,pair_trends,step`,
the step empty on the first), logs
`perturb threshold run=%s fraction=%s holdouts=%d steps=%d percentile=%d delta=%.6f`, at WARNING
under `MIN_STEPS` steps, and makes no API call. The delta logged is what
`bko.evaluate.perturb.DELTAS` records for the run's removal fraction. ADR
`research/identifier-perturbation` records the check.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from bko.datasets.split import split
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import DIRECT_ARM, Experiment
from bko.evaluate.membership import read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import POSITIVE, quality, unmapped
from bko.evaluate.prepare import prepared, split_line
from bko.evaluate.rows_read import frame_of


logger = logging.getLogger("bko.evaluate")

PERCENTILE: Final = 95
"""The percentile of the steps between consecutive holdouts' shapes that delta is."""

MIN_STEPS: Final = 10
"""The fewest steps whose percentile the threshold line logs at INFO; fewer log it at WARNING."""


def threshold(shapes: Sequence[float]) -> float:
    """The `PERCENTILE`th percentile of `|shapes[h] - shapes[h + 1]|`, linearly interpolated.

    Raises:
        EvaluationError: If fewer than 2 shapes leave no step, or a shape is not finite.
    """
    if len(shapes) < 2:
        msg = f"{len(shapes)} holdouts leave no step between consecutive holdouts"
        raise EvaluationError(msg)
    values = np.asarray(shapes, dtype=float)
    if not np.isfinite(values).all():
        msg = f"the shapes {list(shapes)} hold a value that is not finite"
        raise EvaluationError(msg)
    return float(np.percentile(np.abs(np.diff(values)), PERCENTILE))


def perturb_threshold(
    run: str,
    stores: Sequence[Path],
    *,
    experiment: Experiment,
    data: Data,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> str:
    """Read `run`'s `bko-direct` shapes from its `stores`, log delta, and return the CSV.

    Raises:
        EvaluationError: If the run lacks `bko-direct`, or as `threshold`.
        IdentityError: As `bko.evaluate.membership.read`, or if a holdout's `split` line is not
            the one its data prepares, or as `frame_of`, naming the store.
    """
    if DIRECT_ARM not in experiment.arms:
        msg = f"run {run!r} has no {DIRECT_ARM} arm; its arms are {list(experiment.arms)}"
        raise EvaluationError(msg)
    frame = data.frame
    fields = unmapped(data.field_rows, frame=frame)
    held = read(run, stores, data, experiment=experiment, runs=runs, root=root)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["holdout", "shapes", "pair_trends", "step"])
    shapes: list[float] = []
    for seed in experiment.holdout_seeds:
        store = held[seed]
        ready = prepared(
            split(data.records, seed=seed, edges=frame.edges), experiment, edges=frame.edges
        )
        if split_line(ready.ablated) != store.folder.splits[seed]:
            msg = f"holdout {seed}'s split {split_line(ready.ablated)!r} is not"
            msg += f" {store.folder.path}'s, which the store {store.path} is bound to"
            raise IdentityError(msg)
        try:
            rows = frame_of(
                store.rows.lines[seed, DIRECT_ARM],
                arm_line=store.folder.arms[seed, DIRECT_ARM],
                frame=frame,
            )
        except IdentityError as error:
            msg = f"the store {store.path}: {error}"
            raise IdentityError(msg) from error
        train = ready.ablated.train
        found, trends = quality(
            rows, train[train[frame.label] == POSITIVE], fields=fields, frame=frame
        )
        step = "" if not shapes else f"{abs(found - shapes[-1]):.6f}"
        writer.writerow([seed, f"{found:.6f}", f"{trends:.6f}", step])
        logger.info(
            "perturb threshold holdout=%d shapes=%.6f pair_trends=%.6f", seed, found, trends
        )
        shapes.append(found)
    delta = threshold(shapes)
    logger.log(
        logging.WARNING if len(shapes) - 1 < MIN_STEPS else logging.INFO,
        "perturb threshold run=%s fraction=%s holdouts=%d steps=%d percentile=%d delta=%.6f",
        run,
        "none" if experiment.removal is None else experiment.removal.fraction,
        len(shapes),
        len(shapes) - 1,
        PERCENTILE,
        delta,
    )
    return out.getvalue()


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `perturb-threshold` subcommand to `commands`."""
    reading = commands.add_parser(
        "perturb-threshold", help="read the perturbation check's delta from a run's stores"
    )
    reading.add_argument("run", help="the named run whose stored bko-direct rows are read")
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
