"""Each privacy run family's reference arm, which every other arm's privacy is read against.

`reference_of` finds a run's family, by its own entry or the entries of the runs it slices, and
returns the family's reference arm with the cell its experiment removes. ADR
`research/membership-inference-reading` records the rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Stratum
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, REAL_ARM, Experiment
from bko.evaluate.named_runs import NamedRun, wholes_of


REFERENCE: Final[Mapping[str, str]] = MappingProxyType(
    {"e6-thin": REAL_ARM, "e6-empty": REAL_ARM, "e6-curated": REAL_ARM}
    | {"e12-thin": REAL_ARM, "e12-empty": REAL_ARM, "e13": REAL_ARM}
    | {"e14-thin": DIRECT_ARM, "e14-empty": DIRECT_ARM}
    | {
        f"e10-{family}-{cell}": DIRECT_ARM
        for family in ("pilot", "pram")
        for cell in ("thin", "empty")
    }
)
"""Each run family's reference arm, which every other arm's privacy is read against: `real`, or
`bko-direct` for the perturbation runs, whose arms perturb its rows, and for the donor-limit
runs, whose arm limits who donates them."""


def reference_of(
    run: str, experiment: Experiment, runs: Mapping[str, NamedRun]
) -> tuple[str, Stratum]:
    """The reference arm of `run`'s family, and the cell its experiment removes.

    The family is `run`'s own `REFERENCE` entry, or else the entries of the runs it slices,
    which must name one arm.

    Raises:
        EvaluationError: If the family has no reference arm or names several, the experiment
            lacks the reference arm, or the experiment removes no cell, which leaves no cell scope.
    """
    families = {run} if run in REFERENCE else wholes_of(run, runs) & REFERENCE.keys()
    references = {REFERENCE[family] for family in families}
    if not references:
        msg = f"run {run!r} has no reference arm; the families read are {sorted(REFERENCE)}"
        raise EvaluationError(msg)
    if len(references) > 1:
        msg = f"run {run!r} slices {sorted(families)}, whose reference arms differ"
        msg += f": {sorted(references)}"
        raise EvaluationError(msg)
    (reference,) = references
    if reference not in experiment.arms:
        msg = f"run {run!r} lacks its reference arm {reference!r}; its arms are"
        msg += f" {list(experiment.arms)}"
        raise EvaluationError(msg)
    if experiment.removal is None:
        msg = f"run {run!r} removes no cell, so its attack has no cell scope"
        raise EvaluationError(msg)
    return reference, experiment.removal.cell
