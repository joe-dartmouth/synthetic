"""`python -m bko.evaluate panel`: score a recorded run's stored fills under the classifier panel.

`panel` rescores a folder through `bko.evaluate.shared.rescore` at `bko.evaluate.classifier.PANEL`
with the calibration measures, and makes no API call. Each BKO arm is a `StoredArm`, rebuilt from
the folder's candidates file; `none` and `class-weight`, which add no rows, are regenerated; every
other arm is a `StoredBaseline`, answered from the `rows` store. After a holdout passes identity
checks 3 and 4, `bko.evaluate.rows.compare_holdout` holds it to the strict check 7 against the
folder's `scores.csv`, admitting only the panel models' and the calibration measures (`admitted`),
so any moved regression or boosting value exits 3. The whole reading runs on one thread for every
thread-pool API. ADR `research/classifier-panel` records the panel and how it reads a folder.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd
from threadpoolctl import threadpool_limits

from bko.baselines.errors import BaselineError
from bko.baselines.fills import CURRENT_FILL
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, unit_weights
from bko.baselines.request import Method, Request
from bko.evaluate.classifier import MODELS, PANEL, Factory, Model
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import REAL_ARM, Experiment
from bko.evaluate.filling import Logged
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.rows import ScoreKey, compare_holdout, folder_scores, subsampled
from bko.evaluate.rows_read import Rows, bind, frame_of
from bko.evaluate.run_folder import BKO, SCORES_LOG, Folder
from bko.evaluate.scores import Measure, Score, Source
from bko.evaluate.shared import logged_params, rescore


logger = logging.getLogger("bko.evaluate")

REGENERATED: Final = frozenset({Method.NONE.value, Method.CLASS_WEIGHT.value})
"""The arms `panel` runs again: they add no rows, and `class-weight` reaches the model through its
weights, which the store does not hold."""

CALIBRATION: Final = (Measure.BRIER, Measure.CALIBRATION)
"""The calibration measures, under the regression's names; every other model suffixes them."""

ADMITTED: Final = frozenset(
    {
        *(
            Measure(f"{measure}-{model}")
            for model in PANEL
            if model not in MODELS
            for measure in (Measure.RECALL, Measure.PRECISION, Measure.AUC_PR)
        ),
        *(
            measure if model is Model.LR else Measure(f"{measure}-{model}")
            for model in PANEL
            for measure in CALIBRATION
        ),
    }
)
"""The augmented measures `panel` writes beyond a folder scored at `MODELS`."""


def admitted(key: ScoreKey) -> bool:
    """Whether `panel` admits `key` beyond the folder's: a panel model's or calibration measure."""
    return key[2] == Source.AUGMENTED.value and key[3] in ADMITTED


class StoredBaseline:
    """A baseline arm answered from a `rows` store, with no generator run.

    Each fill's attempts before its stored one refuse, so the harness retries them as the run did
    and logs the same `arm` line; the stored attempt returns the stored rows, unit weights and the
    `arm` line's params. A partial fill's stored rows return as a whole fill, so no `shortfall`
    line is logged.
    """

    def __init__(self, name: str, rows: Rows, folder: Folder) -> None:
        """Answer `name`'s gap and replica fills from `rows`, checked against `folder`'s lines."""
        self.name = name
        self._rows = rows
        self._folder = folder
        self._tries: Counter[tuple[int, str]] = Counter()

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """The fill's stored rows on its stored attempt; a refusal before it.

        Raises:
            BaselineError: On an attempt before the stored one.
            IdentityError: If the store or the folder holds no line for the fill, or as
                `frame_of` and `logged_params`.
            EvaluationError: Outside a harness fill.
        """
        fill = CURRENT_FILL.get()
        if fill is None:
            msg = f"the stored baseline {self.name} runs inside a harness fill only"
            raise EvaluationError(msg)
        arm = self.name if fill.kind == "gap" else f"{self.name}-replica"
        key = (fill.holdout, arm)
        self._tries[key] += 1
        line, logged = self._rows.lines.get(key), self._folder.arms.get(key)
        if line is None or logged is None:
            msg = f"holdout {fill.holdout}'s {arm} has no stored fill and arm line in"
            msg += f" {self._folder.path / SCORES_LOG} and its store"
            raise IdentityError(msg)
        if self._tries[key] != line.attempt:
            msg = f"{arm}'s stored attempt {self._tries[key]} did not return; holdout"
            msg += f" {fill.holdout}'s fill returned attempt {line.attempt}"
            raise BaselineError(msg)
        added = frame_of(line, arm_line=logged, frame=request.frame)
        return Baseline(added, unit_weights(train), logged_params(logged))


def runners(
    experiment: Experiment, arms: Mapping[str, Runner], rows: Rows, folder: Folder
) -> dict[str, Runner]:
    """Each non-BKO arm's runner: `arms`' for `REGENERATED`, a `StoredBaseline` for the rest.

    BKO arms are `rescore`'s `StoredArm`s, and `REAL_ARM` has no runner; a `REGENERATED` arm
    `arms` lacks gets none, which `rescore` refuses.
    """
    found: dict[str, Runner] = {}
    for name in experiment.arms:
        if name in BKO or name == REAL_ARM:
            continue
        if name not in REGENERATED:
            found[name] = StoredBaseline(name, rows, folder)
        elif name in arms:
            found[name] = arms[name]
    return found


def panel(
    folder: Folder,
    rows: Rows,
    *,
    run: str,
    experiment: Experiment,
    data: Data,
    arms: Mapping[str, Runner],
    models: Mapping[Model, Factory] = PANEL,
    uci: bool,
) -> str:
    """Score `folder`'s run under `models`, with calibration, from `rows`; return the scores CSV.

    Raises:
        IdentityError: As `bind`, `folder_scores`, `rescore` and `compare_holdout`.
        EvaluationError, BaselineError: As `rescore`.
    """
    bind(rows, folder, data.frame, run=run, data=data.digest)
    theirs = folder_scores(folder)

    def after(holdout: int, logged: Logged, scored: Sequence[Score]) -> None:
        sampled = subsampled(logged.split)
        differing = compare_holdout(
            scored, theirs, holdout=holdout, admitted=admitted, sampled=sampled
        )
        logger.info(
            "panel holdout=%d differing=%d subsampled=%s",
            holdout,
            differing,
            "true" if sampled else "false",
        )

    with threadpool_limits(limits=1):
        return rescore(
            folder,
            [],
            run=run,
            experiment=experiment,
            data=data,
            arms=runners(experiment, arms, rows, folder),
            models=models,
            uci=uci,
            after=after,
            calibration=True,
        )


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `panel` subcommand to `commands`."""
    reading = commands.add_parser(
        "panel", help="score a run's stored fills under the classifier panel, with no verdict"
    )
    reading.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    reading.add_argument("--folder", type=Path, required=True, help="the run's folder under runs/")
    reading.add_argument(
        "--rows", type=Path, required=True, help="the folder's store, which `rows` wrote"
    )
    reading.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
