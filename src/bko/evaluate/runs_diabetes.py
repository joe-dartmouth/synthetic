"""The Diabetes run families: E6's subgroup removal, E9's logic-only ablation, E10's perturbation.

Each experiment builds on `E6_THIN` or `E6_EMPTY`; `_slices` turns one into a named run at
`E6_SETTINGS` and its `E6_SLICES` holdout slices. `bko.evaluate.named_runs.RUNS` names them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    BKO_ARMS,
    CLLM_RAW_ARM,
    CURATED_ARM,
    DIRECT_ARM,
    LOGIC_ARM,
    METHOD_ARMS,
    PERTURBED_ARM,
    PRAM_ARM,
    REAL_ARM,
    SKELETON_ARM,
    Experiment,
    Removal,
)
from bko.evaluate.named_run import (
    BAYES_NET,
    E3,
    NEW_METHODS,
    NamedRun,
)
from bko.generate.messages import SETTINGS


E6_ARMS: Final = (
    *METHOD_ARMS,
    *NEW_METHODS,
    BAYES_NET,
    *BKO_ARMS,
    DIRECT_ARM,
    SKELETON_ARM,
    CLLM_RAW_ARM,
    REAL_ARM,
)
"""E6's arms: every method, every Messages arm but the curated `cllm`, and `real` (ADR
`research/diabetes-subgroup-removal-design`, Decision, Arms)."""


E6_CELL: Final = ("6-7", "male", 1)
"""E6's removed cell: men in their sixties and seventies readmitted within 30 days."""


E6_CAP: Final = 1500.0
"""E6's total spend cap in USD, split between its two fractions' runs."""


E6_ASKED: Final[Mapping[str, int]] = MappingProxyType({"thin": 9_571, "empty": 7_671})
"""Each E6 fraction's estimated records asked per holdout (`docs/EVIDENCE.md`, E6
Configuration)."""


E6_SLICES: Final = 10
"""How many consecutive-holdout slices each E6 fraction's run is cut into."""


E6_SETTINGS: Final = replace(SETTINGS, max_tokens=32000)
"""`SETTINGS` with room for a Diabetes call's reply: at 16000 tokens 1 of 15 pilot calls
stopped at the limit, at 32000 none of 10 did (`docs/EVIDENCE.md`, E6 Configuration)."""


def _e6_ceiling(name: str) -> float:
    """E6 fraction `name`'s share of `E6_CAP`, in proportion to its records asked."""
    return E6_CAP * E6_ASKED[name] / sum(E6_ASKED.values())


E6_THIN: Final = replace(
    E3,
    arms=E6_ARMS,
    candidate="bko",
    comparators=tuple(arm for arm in E6_ARMS if arm not in {"bko", REAL_ARM}),
    replica_arms=("bko",),
    replica_rows=2000,
    removal=Removal(E6_CELL, 0.985),
    fill_to=1269,
    cutoff=0.0902,
    spend_ceiling=_e6_ceiling("thin"),
)
"""E6's thinned run over the Diabetes frame: 0.985 of `E6_CELL`'s 1,269 training rows removed,
each row-adding arm filling it back to 1,269, the training prevalence 0.0902 as the cutoff,
`bko`'s capped replica of 2000 rows, and E3's holdouts, threshold, copy rule and empty criteria
(ADR `research/diabetes-subgroup-removal-design`)."""


E6_EMPTY: Final = replace(
    E6_THIN,
    removal=Removal(E6_CELL, 1.0),
    replica_arms=(),
    replica_rows=None,
    spend_ceiling=_e6_ceiling("empty"),
)
"""E6's zero-shot run: `E6_THIN` with every training row of `E6_CELL` removed and no replica."""


E6_CURATED: Final = replace(
    E6_THIN,
    arms=("none", REAL_ARM, CURATED_ARM),
    candidate=CURATED_ARM,
    comparators=("none",),
    replica_arms=(),
    replica_rows=None,
    spend_ceiling=100.0,
)
"""G7's curated run on `E6_THIN`'s frame, removal and holdouts: `none`, `real` and
`bko-curated`, unsliced, its ceiling $100 (ADR `research/bkb-candidate-curation`; the estimate is
`docs/EVIDENCE.md`, G7). Its splits equal `e6-thin`'s, so `compare --beside` pairs it with `bko`
there."""


E9_THIN, E9_EMPTY = (
    replace(
        e6,
        arms=("none", REAL_ARM, DIRECT_ARM, LOGIC_ARM),
        replica_arms=(),
        replica_rows=None,
        candidate=LOGIC_ARM,
        comparators=("none", DIRECT_ARM),
    )
    for e6 in (E6_THIN, E6_EMPTY)
)
"""E9's runs, `E6_THIN` and `E6_EMPTY` with `none`, `real` and both direct arms, `bko-logic` the
candidate, no replica and no API call (ADR `research/logic-only-ablation`). The spend ceiling
kept from E6 is inert: no arm calls the API."""


E10_PILOT_THIN, E10_PILOT_EMPTY = (
    replace(
        e9,
        arms=("none", REAL_ARM, DIRECT_ARM, PERTURBED_ARM, PRAM_ARM),
        holdout_seeds=(0, 1),
        candidate=PERTURBED_ARM,
        comparators=(DIRECT_ARM, PRAM_ARM, "none"),
        spend_ceiling=5.0,
    )
    for e9 in (E9_THIN, E9_EMPTY)
)
"""E10's pilots: `E9_THIN` and `E9_EMPTY` on holdouts 0 and 1, both perturbation arms in place of
`bko-logic`, the perturbed arm the candidate, at $5.00 (ADR `research/identifier-perturbation`)."""


E10_PRAM_THIN, E10_PRAM_EMPTY = (
    replace(
        e9,
        arms=("none", REAL_ARM, DIRECT_ARM, PRAM_ARM),
        candidate=PRAM_ARM,
        comparators=(DIRECT_ARM, "none"),
    )
    for e9 in (E9_THIN, E9_EMPTY)
)
"""E10's PRAM-only full runs: `E9_THIN` and `E9_EMPTY` on E6's holdouts with `bko-direct-pram` in
place of `bko-logic` as the candidate, and no API call, so the ceiling kept from E6 is inert
(ADR `research/identifier-perturbation`, Full runs)."""


def _slices(
    prefix: str, name: str, whole: Experiment, dataset: Dataset = Dataset.DIABETES
) -> dict[str, NamedRun]:
    """Run `<prefix>-<name>` on `dataset` at `E6_SETTINGS` and its `E6_SLICES` slices.

    `whole` is the run's experiment; `dataset` is Diabetes 130's frame or another over its cohort.

    Each slice takes the next tenth of the holdouts in order, at that share of the ceiling.

    Raises:
        ValueError: If `E6_SLICES` does not divide the holdouts, which would drop the last ones.
    """
    size, left = divmod(len(whole.holdout_seeds), E6_SLICES)
    if left:
        msg = f"{len(whole.holdout_seeds)} holdouts do not divide into {E6_SLICES} slices"
        raise ValueError(msg)
    runs = {f"{prefix}-{name}": NamedRun(whole, E6_SETTINGS, dataset)}
    for index, letter in enumerate("abcdefghij"[:E6_SLICES]):
        sliced = replace(
            whole,
            holdout_seeds=whole.holdout_seeds[index * size : (index + 1) * size],
            spend_ceiling=whole.spend_ceiling / E6_SLICES,
        )
        runs[f"{prefix}-{name}{letter}"] = NamedRun(sliced, E6_SETTINGS, dataset)
    return runs
