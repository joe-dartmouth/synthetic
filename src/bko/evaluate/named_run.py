"""`NamedRun`, one run fixed in code, and the experiments both the run table and its families read.

`E3` is the Synthea removal experiment that E6's runs build on; `NEW_METHODS` and `BAYES_NET` are
arms beside the comparison set. `bko.evaluate.named_runs.RUNS` holds every named run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from bko.baselines.request import Method
from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    ADDS_ROWS,
    BKO_ARMS,
    E1,
    METHOD_ARMS,
    REAL_ARM,
    Experiment,
    Removal,
)
from bko.generate.messages import SETTINGS, Settings


E3: Final = replace(
    E1,
    arms=(*METHOD_ARMS, *BKO_ARMS, REAL_ARM),
    replica_arms=(*ADDS_ROWS, *BKO_ARMS),
    spend_ceiling=65.0,
    criteria=(),
    removal=Removal((">=65", "female", 1), 0.9),
    decontaminate=False,
)
"""E1's arms and `real` over the Synthea frame with 0.9 of `(>=65, female, 1)`'s training rows
removed, test copies kept, a $65.00 ceiling and no criterion (ADR
`research/synthea-downstream-design`)."""


NEW_METHODS: Final = (str(Method.GAUSSIAN_COPULA), str(Method.CART))
"""The classical synthesizers B1 reads beside the comparison set (`docs/EVIDENCE.md`, B1)."""


BAYES_NET: Final = str(Method.BAYES_NET)
"""The Bayesian network baseline's arm (ADR `research/bayesian-network-baseline`)."""


@dataclass(frozen=True, slots=True)
class NamedRun:
    """One run of the harness, fixed in code and selected by name.

    Attributes:
        experiment: What the run fixes before it starts.
        settings: How its generator calls are made; the call log records them.
        dataset: The data it scores (`bko.evaluate.data.load`).
    """

    experiment: Experiment
    settings: Settings = SETTINGS
    dataset: Dataset = Dataset.UCI
