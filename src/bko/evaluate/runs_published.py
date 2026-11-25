"""The experiments run beside published numbers: P1 and P2.

P1 is TableShift's readmission partition, and P2 UCI Heart Disease under SynthEval's metrics.
`bko.evaluate.named_runs.RUNS` names P1's two runs, `p1-id` and `p1-ood`, one per test split
(ADR `research/tableshift-comparison`; `docs/EVIDENCE.md`, P1), and P2's run `p2` (ADR
`research/syntheval-reading`; `docs/EVIDENCE.md`, P2).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from bko.baselines.request import Method
from bko.evaluate.experiment import DIRECT_ARM, E1, METHOD_ARMS
from bko.evaluate.named_run import BAYES_NET, NEW_METHODS


P1_ARMS: Final = (
    *METHOD_ARMS,
    *NEW_METHODS,
    BAYES_NET,
    str(Method.TABDDPM),
    str(Method.LABEL_BOOTSTRAP),
    DIRECT_ARM,
)
"""P1's arms: the comparison set, the classical synthesizers, the Bayesian network, TabDDPM, the
label bootstrap and `bko-direct`."""

P1: Final = replace(
    E1,
    holdout_seeds=tuple(range(10)),
    arms=P1_ARMS,
    candidate=DIRECT_ARM,
    comparators=tuple(arm for arm in P1_ARMS if arm != DIRECT_ARM),
    replica_arms=tuple(arm for arm in P1_ARMS if arm == DIRECT_ARM or Method(arm).adds_rows),
    criteria=(),
    decontaminate=False,
)
"""E1 on ten holdouts with `P1_ARMS`, `bko-direct` the candidate against every other arm, a
replica for every row-adding arm and the candidate, no criterion, and test copies kept, since the
partition is TableShift's. Every holdout scores the same test records after the same training
records (`bko.evaluate.data.held_split`), so its intervals read the arms' draw variance only, and
its held-out ratio is 0 (`bko.evaluate.prepare.held_out_ratio`): each interval is the plain t
interval, with no Nadeau and Bengio test-overlap term."""

P2: Final = replace(
    E1,
    arms=P1_ARMS,
    candidate=DIRECT_ARM,
    comparators=tuple(arm for arm in P1_ARMS if arm != DIRECT_ARM),
    replica_arms=tuple(arm for arm in P1_ARMS if arm == DIRECT_ARM or Method(arm).adds_rows),
    criteria=(),
)
"""E1 on its own holdouts and UCI Heart Disease with `P1_ARMS`, `bko-direct` the candidate against
every other arm, a replica for every row-adding arm and the candidate, no criterion, and E1's
`decontaminate` true: each holdout's test split loses the records some arm's rows copy. Its
replica fills are what `syntheval` scores (`bko.evaluate.syntheval_reading`), against that
decontaminated test split."""
