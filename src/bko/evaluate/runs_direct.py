"""The direct-method study's runs: the control runs on E6's and E3's frames, and its variants.

`E12_THIN` and `E12_EMPTY` are E6's runs with `CONTROL_ARMS`, and `E13` is E3's with them and
`bko-logic`; `E14_THIN` and `E14_EMPTY` hold `bko-capped`, `E15_THIN` and `E15_EMPTY` `bko-tree`,
`E16_THIN` and `E16_EMPTY` the three `bko-hier` arms, `E17_THIN` and `E17_EMPTY`
`bko-coherent`, `E18` the synthetic-share sweep, and `E19_THIN`, `E19_EMPTY`, `E19_SUPPORTED_THIN`
and `E19_SUPPORTED_EMPTY` the matched knowledge controls. `bko.evaluate.named_runs.RUNS` names them
(`docs/EVIDENCE.md`, E12 to E19).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Method
from bko.evaluate.direct_arms import (
    CAPPED_ARM,
    COHERENT_ARM,
    DIRECT_ARM,
    DONOR_ARM,
    HIER_0_ARM,
    HIER_CLOSED_ARM,
    HIER_W_ARM,
    KB_ARM,
    LOGIC_ARM,
    NOMAP_ARM,
    POOL_ALL_ARM,
    POOL_RANDOM_ARM,
    TREE_ARM,
)
from bko.evaluate.experiment import METHOD_ARMS, PRIOR_SHIFT_ARM, REAL_ARM, Experiment
from bko.evaluate.named_run import BAYES_NET, E3
from bko.evaluate.runs_diabetes import E6_CELL, E6_EMPTY, E6_THIN, E9_EMPTY, E9_THIN


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


E14_THIN: Final = replace(
    E9_THIN,
    arms=("none", REAL_ARM, DIRECT_ARM, CAPPED_ARM),
    candidate=CAPPED_ARM,
    comparators=(DIRECT_ARM, "none"),
)
E14_EMPTY: Final = replace(
    E9_EMPTY,
    arms=("none", REAL_ARM, DIRECT_ARM, CAPPED_ARM),
    candidate=CAPPED_ARM,
    comparators=(DIRECT_ARM, "none"),
)
"""E14's runs: `E9_THIN` and `E9_EMPTY` with `bko-capped` in place of `bko-logic` as the candidate
against `bko-direct` and `none`, and no replica, since a replica of every cell would exhaust a
cap of 1 (ADR `research/sampler-donor-limits`). No arm calls the API, so the ceiling is inert."""


E15_THIN, E15_EMPTY = (
    replace(
        e9,
        arms=("none", REAL_ARM, DIRECT_ARM, NOMAP_ARM, TREE_ARM),
        candidate=TREE_ARM,
        comparators=(DIRECT_ARM, NOMAP_ARM, "none"),
    )
    for e9 in (E9_THIN, E9_EMPTY)
)
"""E15's runs: `E9_THIN` and `E9_EMPTY` with `bko-nomap` beside `bko-direct` and `bko-tree` in
place of `bko-logic` as the candidate, and no replica (ADR `research/ancestor-shrunk-shares`).
`bko.evaluate.named_runs.RUNS` scores them on `diabetes-codes`. No arm calls the API, so the
ceiling is inert."""


E16_THIN, E16_EMPTY = (
    replace(
        e9,
        arms=("none", REAL_ARM, DIRECT_ARM, HIER_0_ARM, HIER_W_ARM, HIER_CLOSED_ARM),
        candidate=HIER_W_ARM,
        comparators=(DIRECT_ARM, HIER_0_ARM, "none"),
    )
    for e9 in (E9_THIN, E9_EMPTY)
)
"""E16's runs: `E9_THIN` and `E9_EMPTY` with the three `bko-hier` arms in place of `bko-logic`,
`bko-hier-w` the candidate against `bko-direct`, `bko-hier-0` and `none`, and no replica (ADR
`research/bkb-direct-generation`, Decision 12). No arm calls the API, so the ceiling is inert."""


E17_THIN, E17_EMPTY = (
    replace(
        e9,
        arms=("none", REAL_ARM, DIRECT_ARM, COHERENT_ARM),
        candidate=COHERENT_ARM,
        comparators=(DIRECT_ARM, "none"),
    )
    for e9 in (E9_THIN, E9_EMPTY)
)
"""E17's runs: `E9_THIN` and `E9_EMPTY` with `bko-coherent` in place of `bko-logic` as the candidate
against `bko-direct` and `none`, and no replica (ADR `research/free-field-synthesis`). No arm calls
the API, so the ceiling is inert."""


E18_SHARES: Final[Mapping[str, Fraction]] = MappingProxyType(
    {
        "two": Fraction(1, 50),
        "ten": Fraction(1, 10),
        "quarter": Fraction(1, 4),
        "half": Fraction(1, 2),
        "all": Fraction(1),
    }
)
"""The synthetic shares of E18, by the run name each is spelled as: 2%, 10%, 25%, 50% and 100%."""

E18_ARMS: Final[tuple[str, ...]] = (
    REAL_ARM,
    DIRECT_ARM,
    str(Method.CTGAN),
    str(Method.TVAE),
    str(Method.CART),
    BAYES_NET,
    str(Method.TABDDPM),
    str(Method.LABEL_BOOTSTRAP),
)
"""E18's arms: `real`, `bko-direct`, the row-adding comparators and the label bootstrap, the control
for synthetic rows adding nothing."""

E18: Final[Mapping[str, Experiment]] = MappingProxyType(
    {
        name: replace(
            E6_THIN,
            arms=E18_ARMS,
            candidate=DIRECT_ARM,
            comparators=tuple(arm for arm in E18_ARMS if arm not in {DIRECT_ARM, REAL_ARM}),
            replica_arms=(),
            replica_rows=None,
            removal=None,
            flagged_cell=E6_CELL,
            synthetic_share=share,
        )
        for name, share in E18_SHARES.items()
    }
)
"""One experiment per share: `E6_THIN` with `E18_ARMS`, `bko-direct` the candidate against every
arm but `real`, no replica and no removal, `share` of each cell's training rows replaced by each
arm's fill and `E6_CELL` flagged by name, whose recall the margin reads (ADR
`research/synthetic-share-ablation`). No arm calls the API, so the ceiling is inert."""


E19_THIN_ARMS: Final[tuple[str, ...]] = (
    str(Method.NONE),
    REAL_ARM,
    DIRECT_ARM,
    NOMAP_ARM,
    DONOR_ARM,
    POOL_RANDOM_ARM,
    POOL_ALL_ARM,
    str(Method.ROW_RESAMPLE),
    str(Method.MARGINALS),
    str(Method.EB_MARGINALS),
    str(Method.HB_MARGINALS),
    str(Method.LABEL_BOOTSTRAP),
    PRIOR_SHIFT_ARM,
)
"""`e19-thin`'s arms: `bko-direct`, its matched controls and the multilevel competitor (ADR
`research/matched-knowledge-controls`, Decision 12)."""

E19_EMPTY_ARMS: Final[tuple[str, ...]] = (
    str(Method.NONE),
    REAL_ARM,
    DIRECT_ARM,
    NOMAP_ARM,
    str(Method.MARGINALS),
    str(Method.HB_MARGINALS),
    str(Method.LABEL_BOOTSTRAP),
    PRIOR_SHIFT_ARM,
)
"""`e19-empty`'s arms: `E19_THIN_ARMS` less the five that draw another arm's distribution when the
cell holds no record."""

E19_SUPPORTED_ARMS: Final[tuple[str, ...]] = (
    str(Method.NONE),
    REAL_ARM,
    DIRECT_ARM,
    NOMAP_ARM,
    LOGIC_ARM,
    KB_ARM,
    str(Method.HB_MARGINALS),
    str(Method.LABEL_BOOTSTRAP),
)
"""The supported runs' arms: `bko-kb`, which refuses a frame with a free field, beside `bko-logic`
and the controls that run on `diabetes-supported`."""

E19_THIN: Final = _controls(E6_THIN, E19_THIN_ARMS)
E19_EMPTY: Final = _controls(E6_EMPTY, E19_EMPTY_ARMS)
E19_SUPPORTED_THIN, E19_SUPPORTED_EMPTY = (
    _controls(e6, E19_SUPPORTED_ARMS) for e6 in (E6_THIN, E6_EMPTY)
)
"""E19's runs: E12's runs with each family's arms, `bko-direct` the candidate against every arm but
`real`, and no replica. `bko.evaluate.named_runs.RUNS` scores the supported runs on
`diabetes-supported`. No arm calls the API, so the ceiling is inert."""
