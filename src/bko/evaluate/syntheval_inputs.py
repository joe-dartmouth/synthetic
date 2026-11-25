"""What `syntheval` hands SynthEval: each replica fill with its holdout's splits, and the metrics.

`fills` reads each replica fill of a bound `rows` store with the training split of its holdout
and the test split the run scored, and `METRICS` is the metric set, the one home of the keyword
dictionary passed to `bench/syntheval/score.py` as JSON. `export` writes one fill's inputs into a
directory and `score_argv` names the script's call on them. ADR `research/syntheval-reading`
records the choices; ADR `tooling/benchmark-tools-outside-the-lock` the environment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.evaluate.copies import decontaminated
from bko.evaluate.data import Data, held_split
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import Experiment
from bko.evaluate.prepare import prepared, split_line
from bko.evaluate.rows_read import Rows, frame_of
from bko.evaluate.run_folder import Folder


METRICS: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "pca": {},
        "h_dist": {},
        "corr_diff": {"mixed_corr": True},
        "auroc_diff": {"model": "rf_cls"},
        "cls_acc": {"F1_type": "macro"},
        "eps_risk": {},
        "dcr": {},
        "mia": {"num_eval_iter": 5},
    }
)
"""The metric set: the keyword dictionary Lautrup et al.'s code passes to `SynthEval.evaluate` at
commit `a07cd9c007b58bee37b0837cfdc65252303afadc` (`experiments/auxiliaries/03_baselines.py`)."""

REPLICA: Final = "-replica"
"""The suffix of a replica fill's arm."""

INPUTS: Final = ("real.csv", "test.csv", "synthetic.csv", "metrics.json")
"""The files `export` writes, in the order it writes them."""

OUT: Final = "out.json"
"""The file the script writes its metrics to, beside its inputs."""


@dataclass(frozen=True, slots=True, eq=False)
class SynthevalFill:
    """One replica fill and the splits it is scored against.

    Attributes:
        holdout: The holdout's seed.
        arm: The `arm` line's arm, `<arm>-replica`.
        real: The holdout's training split, in the frame's columns.
        test: The test split the run scored, in the frame's columns: without each record some
            fill of the holdout copies when the experiment decontaminates.
        synthetic: The fill's rows, read from the store.
    """

    holdout: int
    arm: str
    real: pd.DataFrame
    test: pd.DataFrame
    synthetic: pd.DataFrame


def fills(
    folder: Folder, rows: Rows, data: Data, experiment: Experiment
) -> Iterator[SynthevalFill]:
    """Each replica fill of `rows`, holdout by holdout in the experiment's order, arms in order.

    `rows` must already be bound to `folder` (`bko.evaluate.rows_read.bind`). Each holdout's
    split is prepared as the run prepared it and held to the folder's `split` line. When the
    experiment decontaminates (`Experiment.decontaminate`), the test split loses each record some
    stored fill of the holdout, gap or replica, copies (`bko.evaluate.copies.decontaminated`), as
    the run's scoring did.

    Raises:
        IdentityError: If the folder holds no `split` line for a holdout or another one, if a
            replica arm of the experiment has no stored fill and `arm` line for a holdout, or as
            `frame_of`.
    """
    frame = data.frame
    columns = list(frame.columns)
    for seed in experiment.holdout_seeds:
        if seed not in folder.splits:
            msg = f"{folder.path} holds no split line for holdout {seed}"
            raise IdentityError(msg)
        ready = prepared(held_split(data, seed), experiment, edges=frame.edges)
        line = split_line(ready.ablated)
        if line != folder.splits[seed]:
            msg = f"holdout {seed}'s split {line!r} is not {folder.path}'s {folder.splits[seed]!r}"
            raise IdentityError(msg)
        arms = [f"{name}{REPLICA}" for name in experiment.replica_arms]
        for arm in arms:
            if (seed, arm) not in rows.lines or (seed, arm) not in folder.arms:
                msg = f"holdout {seed}'s {arm} has no stored fill and arm line in {folder.path}"
                raise IdentityError(msg)
        stored = {
            arm: frame_of(rows.lines[held, arm], arm_line=folder.arms[held, arm], frame=frame)
            for held, arm in sorted(rows.lines)
            if held == seed
        }
        test = ready.ablated.test
        if experiment.decontaminate:
            test = decontaminated(test, stored.values(), frame=frame)
        real = ready.ablated.train[columns]
        for arm in arms:
            yield SynthevalFill(seed, arm, real, test[columns], stored[arm])


def export(fill: SynthevalFill, directory: Path) -> dict[str, str]:
    """Write `fill`'s three tables and `METRICS` into `directory`; return each file's sha256.

    The tables are CSV without their index; `metrics.json` is `METRICS` with sorted keys.
    """
    texts = {
        "real.csv": fill.real.to_csv(index=False, lineterminator="\n"),
        "test.csv": fill.test.to_csv(index=False, lineterminator="\n"),
        "synthetic.csv": fill.synthetic.to_csv(index=False, lineterminator="\n"),
        "metrics.json": json.dumps({k: dict(v) for k, v in METRICS.items()}, sort_keys=True),
    }
    found = {}
    for name in INPUTS:
        data = texts[name].encode()
        (directory / name).write_bytes(data)
        found[name] = hashlib.sha256(data).hexdigest()
    return found


def score_argv(
    *,
    python: Path,
    script: Path,
    inputs: Path,
    target: str,
    categorical: Sequence[str],
    seed: int,
) -> list[str]:
    """The call of `script` under `python` on the files `export` wrote into `inputs`.

    `seed` seeds the script's global random state; the reading passes the fill's holdout seed.
    """
    return [
        str(python),
        str(script),
        "--real",
        str(inputs / "real.csv"),
        "--test",
        str(inputs / "test.csv"),
        "--synthetic",
        str(inputs / "synthetic.csv"),
        "--target",
        target,
        "--categorical",
        *categorical,
        "--metrics",
        str(inputs / "metrics.json"),
        "--seed",
        str(seed),
        "--out",
        str(inputs / OUT),
    ]
