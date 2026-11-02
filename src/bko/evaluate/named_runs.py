"""Every named run of the harness: its experiment, its settings and the dataset it scores.

`RUNS` holds `e1`, G3's guidance-level runs, `g3-<level>`, one per level, `g3-summary` being E1's
level (`docs/EVIDENCE.md`, G3), the effort sweep's runs, `g4-low` and `g4-medium` (G4), `g2`, the
direct arm's run (`G2`), and `g5`, the prescribing arm's run (`G5`), and E3's runs over the Synthea
frame, `e3` and its holdout slices `e3-a` and `e3-b` (ADR `research/synthea-downstream-design`),
B1's classical-baseline runs, `b1-uci` and `b1-synthea`, B2's Bayesian network runs, `b2-uci` and
`b2-synthea`, and G6's generator-comparison runs,
`g6-sonnet`, `g6-gemma`, `g6-nemotron` and `g6-olmo`, and the two thinking-cost runs,
`g6-gemmathink` and `g6-nemotronthink`.
No command-line flag changes a value here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from bko.baselines.request import Method
from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    ADDS_ROWS,
    BKO_ARMS,
    CLLM_ARM,
    CLLM_RAW_ARM,
    DIRECT_ARM,
    E1,
    HOLDOUTS,
    METHOD_ARMS,
    REAL_ARM,
    SKELETON_ARM,
    Experiment,
    Removal,
)
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

E3_SLICE_CEILING: Final = E3.spend_ceiling / 2
"""Each of E3's two holdout slices' ceiling: half of E3's."""

_E3_SPLIT: Final = HOLDOUTS // 2
"""The first holdout seed of E3's second slice."""


NEW_METHODS: Final = (str(Method.GAUSSIAN_COPULA), str(Method.CART))
"""The classical synthesizers B1 reads beside the comparison set (`docs/EVIDENCE.md`, B1)."""

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

BAYES_NET: Final = str(Method.BAYES_NET)
"""The Bayesian network baseline's arm (ADR `research/bayesian-network-baseline`)."""

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
            f"g4-{effort}": NamedRun(
                SETTINGS_SWEEP,
                Settings(
                    model="claude-haiku-5-5",
                    effort=effort,
                    max_tokens=16000,
                    per_call=20,
                    guidance=Level.SUMMARY,
                ),
            )
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
        "g5": NamedRun(
            G5,
            Settings(
                model="claude-haiku-5-5",
                effort="medium",
                max_tokens=16000,
                per_call=20,
                guidance=Level.SUMMARY,
            ),
        ),
    }
)
"""Every named run, by name: `e1` is `E1` at `SETTINGS`, each `g3-<level>` is E1's methods and
`bko` at that guidance level, judged by no criterion (`docs/EVIDENCE.md`, G3; `g3-summary` is
E1's level), `g4-low` and `g4-medium` are the same experiment at `summary` with effort `low` and
`medium` (G4; `g4-medium`'s settings equal `e1`'s), `g2` is `G2`, and `g5` is `G5` at `summary` with
G4's effort, `medium` (G5), and `e3` is `E3` at `SETTINGS` over the Synthea frame, `e3-a` its
holdouts 0 to 49 and `e3-b` its holdouts 50 to 99, each slice at half its ceiling, `b1-uci` and
`b1-synthea` are `B1_UCI` and `B1_SYNTHEA`, `b2-uci` and `b2-synthea` are `B2_UCI` and
`B2_SYNTHEA`, and each `g6-<model>` is `G6_BASE` at `SETTINGS` with
its `G6_MODELS` model and ceiling (G6), and each `g6-<model>think` is `G6_BASE` at `SETTINGS` with
its `G6_THINKING` model on holdouts 0 to 4 at a $1.00 ceiling. A variant adds its own entry in its
own change, listing its arms, candidate, comparators, holdout seeds, fill target, ceiling and
criteria."""
