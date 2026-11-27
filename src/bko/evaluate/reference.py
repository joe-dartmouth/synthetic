"""Each privacy run family's reference arm, which every other arm's privacy is read against.

`reference_of` finds a run's family, by its own entry or the entries of the runs it slices, and
returns the family's reference arm with the cell its experiment removes, or else flags by name. ADR
`research/membership-inference-reading` records the rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Stratum
from bko.evaluate.direct_arms import CAPPED_ARM
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, REAL_ARM, Experiment
from bko.evaluate.named_runs import NamedRun, wholes_of
from bko.evaluate.runs_direct import E18_SHARES, E21, E22


REFERENCE: Final[Mapping[str, str]] = MappingProxyType(
    {"e6-thin": REAL_ARM, "e6-empty": REAL_ARM, "e6-curated": REAL_ARM}
    | {"e12-thin": REAL_ARM, "e12-empty": REAL_ARM, "e13": REAL_ARM}
    | {"e14-thin": DIRECT_ARM, "e14-empty": DIRECT_ARM}
    | {"e20-thin": CAPPED_ARM, "e20-empty": CAPPED_ARM}
    | {f"e21-{name}": REAL_ARM for name in E21}
    | {"e15-thin": DIRECT_ARM, "e15-empty": DIRECT_ARM}
    | {"e16-thin": DIRECT_ARM, "e16-empty": DIRECT_ARM}
    | {"e17-thin": DIRECT_ARM, "e17-empty": DIRECT_ARM}
    | {f"e18-{share}": DIRECT_ARM for share in E18_SHARES}
    | {f"e22-{name}": DIRECT_ARM for name in E22}
    | {
        f"e19-{family}": REAL_ARM
        for family in ("thin", "empty", "supported-thin", "supported-empty")
    }
    | {
        f"e10-{family}-{cell}": DIRECT_ARM
        for family in ("pilot", "pram")
        for cell in ("thin", "empty")
    }
)
"""Each run family's reference arm, which every other arm's privacy is read against: `real`, as
for the quota sweep, whose arms are E12's, or
`bko-direct` for the perturbation runs and the coherent free-field runs, whose arms perturb its
rows, for the donor-limit, tree-share and conditional `is_a` runs, whose arms change who donates
them or what they draw, and for the synthetic-share runs, whose comparators each read against
`bko-direct`, and for the automatic-mapping runs, whose families change the rows it draws from;
or `bko-capped` for the microaggregation runs, whose candidate `bko-micro` refines
`bko-capped`'s draw, so `bko-micro` and `bko-direct` each read against it."""


def reference_of(
    run: str, experiment: Experiment, runs: Mapping[str, NamedRun]
) -> tuple[str, Stratum]:
    """The reference arm of `run`'s family, and the cell its experiment removes or flags by name.

    The family is `run`'s own `REFERENCE` entry, or else the entries of the runs it slices,
    which must name one arm.

    Raises:
        EvaluationError: If the family has no reference arm or names several, the experiment
            lacks the reference arm, or the experiment neither removes a cell nor flags one by
            name, which leaves no cell scope.
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
    if experiment.removal is not None:
        return reference, experiment.removal.cell
    if experiment.flagged_cell is not None:
        return reference, experiment.flagged_cell
    msg = f"run {run!r} removes no cell and flags none by name, so its attack has no cell scope"
    raise EvaluationError(msg)
