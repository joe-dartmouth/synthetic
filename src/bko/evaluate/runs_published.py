"""The experiments run beside published numbers: P1, P2 and P3.

P1 is TableShift's readmission partition, P2 UCI Heart Disease under SynthEval's metrics, and P3
Juwara et al.'s removal fractions on E6's cell. `bko.evaluate.named_runs.RUNS` names P1's four
runs, `p1-id` and `p1-ood`, one per test split, and their partial-replica runs `p1-id-partial` and
`p1-ood-partial` (ADR `research/tableshift-comparison`; `docs/EVIDENCE.md`, P1), P2's run `p2`
(ADR `research/syntheval-reading`; `docs/EVIDENCE.md`, P2) and P3's runs `p3-<fraction>` with
their slices (ADR `research/covariate-bias-comparison`; `docs/EVIDENCE.md`, P3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Method
from bko.datasets.ablation import removed_count
from bko.evaluate.experiment import (
    DIRECT_ARM,
    E1,
    METHOD_ARMS,
    NOMAP_ARM,
    PRIOR_SHIFT_ARM,
    REAL_ARM,
    Experiment,
    Removal,
)
from bko.evaluate.named_run import BAYES_NET, NEW_METHODS
from bko.evaluate.runs_diabetes import E6_CELL, E6_THIN


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

P1_PARTIAL: Final = replace(P1, partial_replica=True)
"""P1 with `partial_replica` set, as TVAE's replica of TableShift's rare cells needs (ADR
`research/tableshift-comparison`); ADR `research/e1-evaluation-protocol` holds the rule."""

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


P3_FRACTIONS: Final[Mapping[str, float]] = MappingProxyType(
    {"fifteen": 0.15, "thirty": 0.30, "fifty": 0.50, "eighty": 0.80, "ninetyfive": 0.95}
)
"""Juwara et al.'s removal fractions, by the run name each is spelled as: 15%, 30%, 50% and 80% on
their real data and up to 95% in their simulation."""

P3_ARMS: Final = (
    str(Method.NONE),
    REAL_ARM,
    str(Method.SMOTE),
    str(Method.CTGAN),
    str(Method.CART),
    DIRECT_ARM,
    NOMAP_ARM,
    str(Method.LABEL_BOOTSTRAP),
    PRIOR_SHIFT_ARM,
)
"""P3's arms: `none`, `real`, the row-adding comparators `smote`, `ctgan` and `cart` (standing in
for their sequential boosted trees), `bko-direct` and the controls of E12."""


def juwara_total(count: int, fraction: float) -> int:
    """Juwara et al.'s `total_size`: the rows the removal leaves over `1 - fraction`, floored.

    The removal takes `removed_count(fraction, count)` of the cell's `count` rows, as
    `bko.datasets.ablation.thinned` does, and the division is exact.

    Raises:
        ValueError: If `fraction` is not strictly between 0 and 1, so no count can be restored.
    """
    share = Fraction(str(fraction))
    if not 0 < share < 1:
        msg = f"the removal fraction {fraction} must lie strictly between 0 and 1"
        raise ValueError(msg)
    remaining = count - removed_count(fraction, count)
    return math.floor(Fraction(remaining) / (1 - share))


P3: Final[Mapping[str, Experiment]] = MappingProxyType(
    {
        name: replace(
            E6_THIN,
            arms=P3_ARMS,
            candidate=DIRECT_ARM,
            comparators=tuple(arm for arm in P3_ARMS if arm not in {DIRECT_ARM, REAL_ARM}),
            replica_arms=(),
            replica_rows=None,
            removal=Removal(E6_CELL, fraction),
            flagged_cell=E6_CELL,
            fill_to=juwara_total(E6_THIN.fill_to, fraction),
        )
        for name, fraction in P3_FRACTIONS.items()
    }
)
"""One experiment per fraction: `E6_THIN` with `P3_ARMS`, `bko-direct` the candidate against every
arm but `real`, no replica, `fraction` of `E6_CELL`'s training rows removed, that cell flagged by
name and each row-adding arm filling it to `juwara_total` of its training rows, which
`E6_THIN.fill_to` holds (ADR `research/covariate-bias-comparison`). No arm calls the API, so the
ceiling is inert."""
