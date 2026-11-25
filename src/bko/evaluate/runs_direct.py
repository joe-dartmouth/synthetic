"""The direct-method study's runs: the control runs on E6's and E3's frames.

`E12_THIN` and `E12_EMPTY` are E6's runs with `CONTROL_ARMS`, and `E13` is E3's with them and
`bko-logic`; `bko.evaluate.named_runs.RUNS` names them (`docs/EVIDENCE.md`, E12 and E13).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from bko.baselines.request import Method
from bko.evaluate.experiment import (
    DIRECT_ARM,
    LOGIC_ARM,
    METHOD_ARMS,
    NOMAP_ARM,
    PRIOR_SHIFT_ARM,
    REAL_ARM,
    Experiment,
)
from bko.evaluate.named_run import BAYES_NET, E3
from bko.evaluate.runs_diabetes import E6_EMPTY, E6_THIN


CONTROL_ARMS: Final[tuple[str, ...]] = (
    *METHOD_ARMS,
    REAL_ARM,
    BAYES_NET,
    DIRECT_ARM,
    NOMAP_ARM,
    str(Method.LABEL_BOOTSTRAP),
    PRIOR_SHIFT_ARM,
)
"""The control runs' arms: the comparison set, `real`, the Bayesian network, `bko-direct` and the
three controls the control reading rule contrasts it with (`docs/RESEARCH.md`, Control reading
rule)."""


def _controls(experiment: Experiment, arms: tuple[str, ...]) -> Experiment:
    """`experiment` with `arms`, `bko-direct` the candidate against all but `real`, no replica."""
    return replace(
        experiment,
        arms=arms,
        candidate=DIRECT_ARM,
        comparators=tuple(arm for arm in arms if arm not in {DIRECT_ARM, REAL_ARM}),
        replica_arms=(),
        replica_rows=None,
    )


E12_THIN: Final = _controls(E6_THIN, CONTROL_ARMS)
E12_EMPTY: Final = _controls(E6_EMPTY, CONTROL_ARMS)
"""E12's runs: `E6_THIN` and `E6_EMPTY` with `CONTROL_ARMS`, `bko-direct` the candidate and no
replica. No arm calls the API, so the spend ceiling kept from E6 is inert."""


E13: Final = replace(_controls(E3, (*CONTROL_ARMS, LOGIC_ARM)), spend_ceiling=1.0)
"""E13's run: `E3` with `CONTROL_ARMS` and `bko-logic`, `bko-direct` the candidate, no replica and
E3's removal, holdouts and copy rule, so `compare --beside` pairs it with E3's runs. No arm calls
the API, so its $1.00 ceiling is inert."""
