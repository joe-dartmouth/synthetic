"""Every named run of the harness: its experiment, its settings and the dataset it scores.

`RUNS` holds `e1`, G3's guidance-level runs, `g3-<level>`, one per level, `g3-summary` being E1's
level (`docs/EVIDENCE.md`, G3), the effort sweep's runs, `g4-low` and `g4-medium` (G4), `g2`, the
direct arm's run (`G2`), and `g5`, the prescribing arm's run (`G5`), and E3's runs over the Synthea
frame, `e3` and its holdout slices `e3-a` and `e3-b` (ADR `research/synthea-downstream-design`).
No command-line flag changes a value here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    ADDS_ROWS,
    BKO_ARMS,
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
        "g2": NamedRun(G2),
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
holdouts 0 to 49 and `e3-b` its holdouts 50 to 99, each slice at half its ceiling. A
variant adds its own entry in its own change, listing its arms, candidate, comparators, holdout
seeds, fill target,
ceiling and criteria."""
