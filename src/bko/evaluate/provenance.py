"""What a `score` or `contradictions` run logs about the code and libraries that produced it.

The `versions` line itself is `bko.core.versions`', re-exported here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from bko.core.versions import VERSIONED, commit, log_versions, versions
from bko.datasets.split import share_text
from bko.evaluate.experiment import Experiment
from bko.evaluate.utility import THRESHOLD
from bko.gaps.quota import QuotaRule


logger = logging.getLogger("bko.evaluate")

__all__ = [
    "VERSIONED",
    "commit",
    "experiment_line",
    "log_experiment",
    "log_versions",
    "versions",
]
"""`bko.core.versions`' names stay importable here, where the stages first found them."""


def log_experiment(name: str, experiment: Experiment, models: Iterable[str], *, frame: str) -> None:
    """Log the run's `experiment_line`, then the commit and libraries it ran with."""
    logger.info("%s", experiment_line(name, experiment, models, frame=frame))
    log_versions()


def experiment_line(name: str, experiment: Experiment, models: Iterable[str], *, frame: str) -> str:
    """The run's configuration, frame and models, as the `experiment` log line's message.

    `seeds=` is the first and last holdout seed, `<first>-<last>`; `removal=` the removed cell's
    band, sex and label joined by `/`, then `@` and the fraction, or `none`; `flagged_cell=`
    follows `removal=` only when set, as the cell's band, sex and label joined by `/`; `cutoff=`
    follows `fill_to=` only when it is not E1's; `quota_rule=` follows `fill_to=` and `cutoff=` only
    when it is not `fill-to`; `replica_rows=` follows `replica=` only when set;
    `synthetic_share=` follows `training_fraction=` only when set, as `<numerator>/<denominator>`.
    """
    seeds, removal = experiment.holdout_seeds, experiment.removal
    removed = (
        "none" if removal is None else f"{'/'.join(map(str, removal.cell))}@{removal.fraction}"
    )
    cell = experiment.flagged_cell
    named = "" if cell is None else f"flagged_cell={'/'.join(map(str, cell))} "
    cutoff = "" if experiment.cutoff == THRESHOLD else f"cutoff={experiment.cutoff} "
    share = experiment.synthetic_share
    shared = "" if share is None else f"synthetic_share={share_text(share)} "
    capped = "" if experiment.replica_rows is None else f"replica_rows={experiment.replica_rows} "
    rule = experiment.quota_rule
    ruled = "" if rule is QuotaRule.FILL_TO else f"quota_rule={rule} "
    return (
        f"experiment run={name} frame={frame} holdouts={len(seeds)} seeds={seeds[0]}-{seeds[-1]} "
        f"removal={removed} {named}training_fraction={experiment.training_fraction} {shared}"
        f"decontaminate={'true' if experiment.decontaminate else 'false'} "
        f"arms={','.join(experiment.arms)} replica={','.join(experiment.replica_arms)} {capped}"
        f"threshold={experiment.threshold} fill_to={experiment.fill_to} {cutoff}{ruled}"
        f"budget={experiment.budget} validity={experiment.validity} "
        f"spend_ceiling={experiment.spend_ceiling:.2f} "
        f"criteria={','.join(experiment.criteria) or 'none'} models={','.join(models)}"
    )
