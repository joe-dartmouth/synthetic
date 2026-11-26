"""Every named run of the harness: its experiment, its settings and the dataset it scores.

`RUNS` holds every named run, by name, and its docstring says what each one is.
No command-line flag changes a value here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    ADDS_ROWS,
    CLLM_ARM,
    CLLM_RAW_ARM,
    DIRECT_ARM,
    E1,
    HOLDOUTS,
    METHOD_ARMS,
    REAL_ARM,
    SKELETON_ARM,
)
from bko.evaluate.named_run import (
    BAYES_NET,
    E3,
    NEW_METHODS,
    NamedRun,
)
from bko.evaluate.runs_diabetes import (
    E6_ARMS,  # ruff: ignore[unused-import]  (re-exported)
    E6_ASKED,  # ruff: ignore[unused-import]  (re-exported)
    E6_CAP,  # ruff: ignore[unused-import]  (re-exported)
    E6_CELL,  # ruff: ignore[unused-import]  (re-exported)
    E6_CURATED,
    E6_EMPTY,
    E6_SETTINGS,
    E6_SLICES,  # ruff: ignore[unused-import]  (re-exported)
    E6_THIN,
    E9_EMPTY,
    E9_THIN,
    E10_PILOT_EMPTY,
    E10_PILOT_THIN,
    E10_PRAM_EMPTY,
    E10_PRAM_THIN,
    _e6_ceiling,  # ruff: ignore[unused-import]  (re-exported)
    _slices,
)
from bko.evaluate.runs_direct import (
    E12_EMPTY,
    E12_THIN,
    E13,
    E14_EMPTY,
    E14_THIN,
    E15_EMPTY,
    E15_THIN,
    E16_EMPTY,
    E16_THIN,
    E17_EMPTY,
    E17_THIN,
    E18,
    E19_EMPTY,
    E19_SUPPORTED_EMPTY,
    E19_SUPPORTED_THIN,
    E19_THIN,
    E20_EMPTY,
    E20_THIN,
)
from bko.evaluate.runs_published import P1, P2, P3
from bko.generate.messages import LEVELS, SETTINGS, Level, Settings


G2: Final = replace(
    E1,
    arms=(*METHOD_ARMS, DIRECT_ARM),
    candidate=DIRECT_ARM,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, DIRECT_ARM),
    criteria=(),
)
"""E1 with the direct arm in place of the BKO arms, judged by no criterion; `docs/EVIDENCE.md`
(G2) names it."""

G5: Final = replace(
    E1,
    arms=(*METHOD_ARMS, "bko", SKELETON_ARM),
    candidate=SKELETON_ARM,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, "bko", SKELETON_ARM),
    spend_ceiling=25.0,
    criteria=(),
)
"""E1 with the prescribing arm beside an in-run `bko`, a $25.00 ceiling and no criterion;
`docs/EVIDENCE.md` (G5) names it."""


SETTINGS_SWEEP: Final = replace(
    E1,
    arms=(*METHOD_ARMS, "bko"),
    replica_arms=(*ADDS_ROWS, "bko"),
    spend_ceiling=10.0,
    criteria=(),
)
"""E1 with its methods and `bko` alone, a $10.00 ceiling and no criterion: the experiment shared by
the runs that vary only the settings, every guidance-level run (`docs/EVIDENCE.md`, G3) and every
effort-sweep run (G4)."""


E4_TARGETS: Final[Mapping[str, int]] = MappingProxyType(
    {"e4-thirty": 30, "e4-forty": 40, "e4-eighty": 80}
)
"""Each dose run's fill target: `SETTINGS_SWEEP` filling every flagged cell past E1's threshold of
20, the threshold unchanged so the same cells are flagged; `docs/EVIDENCE.md` (E4) names them."""


E5_FRACTIONS: Final[Mapping[str, Fraction]] = MappingProxyType(
    {
        "e5-quarter": Fraction(1, 4),
        "e5-half": Fraction(1, 2),
        "e5-threequarters": Fraction(3, 4),
    }
)
"""Each scarcity run's training fraction: E1's methods and `bko` at `SETTINGS`, every holdout
keeping that share of each cell's training rows, a $10.00 ceiling and no criterion;
`docs/EVIDENCE.md` (E5) names them."""


E3_SLICE_CEILING: Final = E3.spend_ceiling / 2
"""Each of E3's two holdout slices' ceiling: half of E3's."""

_E3_SPLIT: Final = HOLDOUTS // 2
"""The first holdout seed of E3's second slice."""


B1_UCI: Final = replace(
    E1,
    arms=(*METHOD_ARMS, *NEW_METHODS),
    candidate=NEW_METHODS[0],
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *NEW_METHODS),
    criteria=(),
)
"""E1's methods and the classical synthesizers, the copula as the candidate, no BKO arm and no
criterion, so it makes no API call (`docs/EVIDENCE.md`, B1)."""

B1_SYNTHEA: Final = replace(
    E3,
    arms=(*METHOD_ARMS, *NEW_METHODS, REAL_ARM),
    candidate=NEW_METHODS[0],
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *NEW_METHODS),
)
"""E3's removal and copy rule with E3's methods, the classical synthesizers and `real`, no BKO arm
(`docs/EVIDENCE.md`, B1)."""


B2_UCI: Final = replace(
    E1,
    arms=(*METHOD_ARMS, BAYES_NET),
    candidate=BAYES_NET,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, BAYES_NET),
    criteria=(),
)
"""E1's methods and the Bayesian network as the candidate, no BKO arm and no criterion, so it makes
no API call (`docs/EVIDENCE.md`, B2)."""

B2_SYNTHEA: Final = replace(
    E3,
    arms=(*METHOD_ARMS, BAYES_NET, REAL_ARM),
    candidate=BAYES_NET,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, BAYES_NET),
)
"""E3's removal and copy rule with E3's methods, the Bayesian network and `real`, no BKO arm
(`docs/EVIDENCE.md`, B2)."""


B3_ARMS: Final = ("bko", CLLM_ARM, CLLM_RAW_ARM)
"""The Messages arms B3 reads together: `bko`, CLLM and its uncurated twin (`docs/EVIDENCE.md`,
B3)."""

B3_UCI: Final = replace(
    E1,
    arms=(*METHOD_ARMS, *B3_ARMS),
    candidate=CLLM_ARM,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *B3_ARMS),
    criteria=(),
    spend_ceiling=20.0,
)
"""E1's methods and B3's arms over UCI, CLLM as the candidate, no criterion, a $20.00 ceiling."""

B3_SYNTHEA: Final = replace(
    E3,
    arms=(*METHOD_ARMS, *B3_ARMS, REAL_ARM),
    candidate=CLLM_ARM,
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *B3_ARMS),
    spend_ceiling=35.0,
)
"""E3's removal and copy rule with E3's methods, B3's arms and `real`, CLLM as the candidate, no
criterion, a $35.00 ceiling."""


G6_BASE: Final = replace(
    E1, arms=(*METHOD_ARMS, "bko"), replica_arms=(*ADDS_ROWS, "bko"), criteria=()
)
"""E1's methods and `bko` at `SETTINGS` but for the model, judged by no criterion: the experiment
every generator-comparison run shares (`docs/EVIDENCE.md`, G6)."""

G6_MODELS: Final[Mapping[str, tuple[str, float]]] = MappingProxyType(
    {
        "g6-sonnet": ("claude-sonnet-5-5", 90.0),
        "g6-gemma": ("gemma-4-26b-a4b-q8_0", 1.0),
        "g6-nemotron": ("nemotron-3-nano-30b-a3b-q8_0", 1.0),
        "g6-olmo": ("olmo-3-7b-instruct-q8_0", 1.0),
    }
)
"""Each generator-comparison run's model and spend ceiling in USD. Sonnet 5.5's covers 1.25 times
its estimate. An on-device model bills nothing, so its ceiling never stops the run and $1.00
serves as any ceiling above 0 would; `bko.evaluate.context.Context.sender` keeps its calls local."""

G6_THINKING: Final[Mapping[str, str]] = MappingProxyType(
    {
        "g6-gemmathink": "gemma-4-26b-a4b-q8_0-thinking",
        "g6-nemotronthink": "nemotron-3-nano-30b-a3b-q8_0-thinking",
    }
)
"""Each thinking-cost run's model: an on-device model's pin that lets it reason before it replies.
Each run is `G6_BASE` on holdouts 0 to 4 at a $1.00 ceiling and an idle limit of 1700 seconds, a
liveness guard below `bko.generate.local.TIMEOUT_SECONDS`, not a measurement."""


G4_SETTINGS: Final = Settings(
    model="claude-haiku-5-5", effort="medium", max_tokens=16000, per_call=20, guidance=Level.SUMMARY
)
"""Haiku 5.5 at `summary` and effort `medium`: G4's medium run's settings and G5's."""

DEFAULT_RUN: Final = "e1"
"""The run `score` and `compare` take when none is named."""

RUNS: Final[Mapping[str, NamedRun]] = MappingProxyType(
    {
        DEFAULT_RUN: NamedRun(E1),
        **{
            f"g3-{level}": NamedRun(
                SETTINGS_SWEEP,
                replace(SETTINGS, guidance=level),
            )
            for level in LEVELS
        },
        **{
            f"g4-{effort}": NamedRun(SETTINGS_SWEEP, replace(G4_SETTINGS, effort=effort))
            for effort in ("low", "medium")
        },
        "e3": NamedRun(E3, SETTINGS, Dataset.SYNTHEA),
        "e3-a": NamedRun(
            replace(E3, holdout_seeds=tuple(range(_E3_SPLIT)), spend_ceiling=E3_SLICE_CEILING),
            SETTINGS,
            Dataset.SYNTHEA,
        ),
        "e3-b": NamedRun(
            replace(
                E3, holdout_seeds=tuple(range(_E3_SPLIT, HOLDOUTS)), spend_ceiling=E3_SLICE_CEILING
            ),
            SETTINGS,
            Dataset.SYNTHEA,
        ),
        **{
            name: NamedRun(replace(SETTINGS_SWEEP, fill_to=target))
            for name, target in E4_TARGETS.items()
        },
        **{
            name: NamedRun(replace(SETTINGS_SWEEP, training_fraction=fraction))
            for name, fraction in E5_FRACTIONS.items()
        },
        "g2": NamedRun(G2),
        "b1-uci": NamedRun(B1_UCI),
        "b1-synthea": NamedRun(B1_SYNTHEA, SETTINGS, Dataset.SYNTHEA),
        "b2-uci": NamedRun(B2_UCI),
        "b2-synthea": NamedRun(B2_SYNTHEA, SETTINGS, Dataset.SYNTHEA),
        "b3-uci": NamedRun(B3_UCI),
        "b3-synthea": NamedRun(B3_SYNTHEA, SETTINGS, Dataset.SYNTHEA),
        **{
            name: NamedRun(replace(G6_BASE, spend_ceiling=ceiling), replace(SETTINGS, model=model))
            for name, (model, ceiling) in G6_MODELS.items()
        },
        **{
            name: NamedRun(
                replace(
                    G6_BASE, holdout_seeds=tuple(range(5)), spend_ceiling=1.0, idle_limit=1700.0
                ),
                replace(SETTINGS, model=model),
            )
            for name, model in G6_THINKING.items()
        },
        **_slices("e6", "thin", E6_THIN),
        **_slices("e6", "empty", E6_EMPTY),
        **_slices("e9", "thin", E9_THIN),
        **_slices("e9", "empty", E9_EMPTY),
        "e6-curated": NamedRun(E6_CURATED, E6_SETTINGS, Dataset.DIABETES),
        "e10-pilot-thin": NamedRun(E10_PILOT_THIN, E6_SETTINGS, Dataset.DIABETES),
        "e10-pilot-empty": NamedRun(E10_PILOT_EMPTY, E6_SETTINGS, Dataset.DIABETES),
        **_slices("e10", "pram-thin", E10_PRAM_THIN),
        **_slices("e10", "pram-empty", E10_PRAM_EMPTY),
        "g5": NamedRun(G5, G4_SETTINGS),
        **_slices("e12", "thin", E12_THIN),
        **_slices("e12", "empty", E12_EMPTY),
        "e13": NamedRun(E13, SETTINGS, Dataset.SYNTHEA),
        **_slices("e14", "thin", E14_THIN),
        **_slices("e14", "empty", E14_EMPTY),
        **_slices("e15", "thin", E15_THIN, Dataset.DIABETES_CODES),
        **_slices("e15", "empty", E15_EMPTY, Dataset.DIABETES_CODES),
        **_slices("e16", "thin", E16_THIN),
        **_slices("e16", "empty", E16_EMPTY),
        **_slices("e17", "thin", E17_THIN),
        **_slices("e17", "empty", E17_EMPTY),
        **{key: run for name in P3 for key, run in _slices("p3", name, P3[name]).items()},
        **{key: run for name in E18 for key, run in _slices("e18", name, E18[name]).items()},
        **_slices("e19", "thin", E19_THIN),
        **_slices("e19", "empty", E19_EMPTY),
        **_slices("e19", "supported-thin", E19_SUPPORTED_THIN, Dataset.DIABETES_SUPPORTED),
        **_slices("e19", "supported-empty", E19_SUPPORTED_EMPTY, Dataset.DIABETES_SUPPORTED),
        **_slices("e20", "thin", E20_THIN),
        **_slices("e20", "empty", E20_EMPTY),
        "p1-id": NamedRun(P1, SETTINGS, Dataset.TABLESHIFT_ID),
        "p1-ood": NamedRun(P1, SETTINGS, Dataset.TABLESHIFT_OOD),
        "p2": NamedRun(P2, SETTINGS),
    }
)
"""Every named run, by family: `e1`, `g2` to `g6-*`, `b1-*` to `b3-*`, `e3` to `e5-*`, the
Diabetes runs `e6-*`, `e9-*`, the perturbation pilots `e10-pilot-*` and the PRAM-only full runs
`e10-pram-*`, the control runs `e12-*` and `e13`, the donor-limit runs `e14-*`, the tree-share
runs `e15-*` on `diabetes-codes`, the conditional `is_a` runs `e16-*`, the coherent free-field
runs `e17-*`, Juwara et al.'s removal fractions `p3-*`, the synthetic-share sweep `e18-*`,
the matched knowledge controls `e19-thin*`, `e19-empty*` and, on `diabetes-supported`,
`e19-supported-*`, the microaggregation runs `e20-*`, TableShift's `p1-id` and `p1-ood`, and
SynthEval's `p2`, each family's experiment saying what it is. A variant adds its own entry in its
own change, with its arms, candidate, comparators and ceiling."""


def sliced(part: NamedRun, whole: NamedRun) -> bool:
    """Whether `part` is a holdout slice of `whole`.

    It is when its settings and dataset are `whole`'s, its holdouts are among `whole`'s, and its
    experiment but for its holdouts and spend ceiling is `whole`'s.
    """
    if (part.settings, part.dataset) != (whole.settings, whole.dataset):
        return False
    rest = replace(
        part.experiment,
        holdout_seeds=whole.experiment.holdout_seeds,
        spend_ceiling=whole.experiment.spend_ceiling,
    )
    held = set(part.experiment.holdout_seeds) <= set(whole.experiment.holdout_seeds)
    return held and rest == whole.experiment


def wholes_of(name: str, runs: Mapping[str, NamedRun] = RUNS) -> frozenset[str]:
    """Every run of `runs` that the run `name` is a holdout slice of; empty for none.

    Only a run with more holdouts than `name`'s counts, so a run is not its own slice. A caller
    checks a slice against the run it names, so no whole is preferred over another.
    """
    part = runs.get(name)
    if part is None:
        return frozenset()
    count = len(part.experiment.holdout_seeds)
    return frozenset(
        other
        for other, whole in runs.items()
        if len(whole.experiment.holdout_seeds) > count and sliced(part, whole)
    )
