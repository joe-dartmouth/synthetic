"""D1's sweep: every gap definition at each grounding count or level, and the level the rule picks.

`docs/EVIDENCE.md` (D1) names the run. A setting is a grounding definition at each count of
`GROUNDING_COUNTS` and the default level, or the frequency-aware definition at each level of
`CREDIBILITY_LEVELS` and count 1, since each definition ignores the other's parameter. `sweep` reads
one E2 ablation at every setting against its truth, as `bko.gaps.experiment` reads the defaults;
`choose` applies the rule ADR `research/frequency-aware-gap-score` states before the run; `listing`
lists a dataset's gaps at every setting, for UCI's training split. The sweep only logs, so the
commands' stdout does not change; no command-line flag changes a value here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from bko.bkb.structure import BKB
from bko.gaps.definitions import CREDIBILITY, Definition, StatementGap, gaps
from bko.gaps.detect import quiet_fusion
from bko.gaps.experiment import AblationRun, Detection, detection, share
from bko.gaps.observe import Cells


logger = logging.getLogger("bko.gaps")

GROUNDING_COUNTS: Final = (1, 2, 3, 5)
"""The grounding counts the per-subgroup and population-level definitions are read at."""

CREDIBILITY_LEVELS: Final = (0.90, 0.95, 0.99)
"""The levels the frequency-aware definition is read at."""

CHOOSING_ABLATION: Final = "cell-50"
"""The ablation whose reading chooses the level: E2's `cell` ablation at fraction 0.5."""


@dataclass(frozen=True, slots=True)
class Setting:
    """One definition at one grounding count and level.

    Attributes:
        definition: The definition.
        grounding: The grounding count `gaps` is called with.
        credibility: The level `gaps` is called with.
    """

    definition: Definition
    grounding: int
    credibility: float


SETTINGS: Final = (
    *(Setting(Definition.SUBGROUP, k, CREDIBILITY) for k in GROUNDING_COUNTS),
    *(Setting(Definition.POPULATION, k, CREDIBILITY) for k in GROUNDING_COUNTS),
    *(Setting(Definition.FREQUENCY, 1, level) for level in CREDIBILITY_LEVELS),
)
"""Every setting, in log order."""


@dataclass(frozen=True, slots=True)
class Reading:
    """One setting's sensitivity and specificity for one ablation.

    Attributes:
        setting: The setting.
        detection: Its reading against the ablation's truth.
    """

    setting: Setting
    detection: Detection


@dataclass(frozen=True, slots=True)
class Sweep:
    """One ablation read at every setting.

    Attributes:
        ablation: The ablation's name.
        count: The count-based detector's reading, the sensitivity a level must reach.
        readings: Each setting's reading, in `SETTINGS` order.
    """

    ablation: str
    count: Detection
    readings: tuple[Reading, ...]

    def detection(self, setting: Setting) -> Detection:
        """`setting`'s reading.

        Raises:
            KeyError: If the sweep did not read `setting`.
        """
        for reading in self.readings:
            if reading.setting == setting:
                return reading.detection
        raise KeyError(setting)


@dataclass(frozen=True, slots=True)
class Choice:
    """The level the rule chooses, with the measures that chose it.

    Attributes:
        ablation: `CHOOSING_ABLATION`, or `None` when the sweeps hold no such ablation.
        level: The chosen level; `None` when no level qualifies.
        floor: The count-based detector's sensitivity on `CHOOSING_ABLATION`, which a level's
            must reach; `None` when the sweeps hold no such ablation or it reads `n/a`.
        sensitivity: The chosen level's sensitivity.
        specificity: The chosen level's specificity.
        tied: Every qualifying level sharing the chosen level's specificity, ascending.
    """

    ablation: str | None
    level: float | None
    floor: float | None
    sensitivity: float | None
    specificity: float | None
    tied: tuple[float, ...]


def sweep(run: AblationRun, fragments: Sequence[BKB]) -> Sweep:
    """Read `run`'s ablation at every setting against its truth.

    Args:
        run: The ablation, read at the defaults.
        fragments: The ontology fragments `run` was read with.
    """
    universe = run.universe
    readings: list[Reading] = []
    for setting in SETTINGS:
        if setting == Setting(setting.definition, 1, CREDIBILITY):
            before, after = run.before[setting.definition], run.after[setting.definition]
        else:
            before, after = (
                _gaps(setting, fragments, cells) for cells in (run.before_cells, run.after_cells)
            )
        readings.append(Reading(setting, detection(before, after, run.truth, universe)))
    return Sweep(run.name, run.count_detection, tuple(readings))


def choose(readings: Iterable[Sweep]) -> Choice:
    """The level the rule of ADR `research/frequency-aware-gap-score` chooses from the sweeps."""
    chosen = next((one for one in readings if one.ablation == CHOOSING_ABLATION), None)
    if chosen is None:
        return Choice(None, None, None, None, None, ())
    floor = chosen.count.sensitivity
    if floor is None:
        return Choice(CHOOSING_ABLATION, None, None, None, None, ())
    qualifying: dict[float, tuple[float, float]] = {}
    for reading in chosen.readings:
        sensitivity, specificity = reading.detection.sensitivity, reading.detection.specificity
        if (
            reading.setting.definition is Definition.FREQUENCY
            and sensitivity is not None
            and specificity is not None
            and sensitivity >= floor
        ):
            qualifying[reading.setting.credibility] = (sensitivity, specificity)
    if not qualifying:
        return Choice(CHOOSING_ABLATION, None, floor, None, None, ())
    best = max(specificity for _, specificity in qualifying.values())
    tied = tuple(sorted(level for level, (_, spec) in qualifying.items() if spec == best))
    return Choice(CHOOSING_ABLATION, tied[0], floor, qualifying[tied[0]][0], best, tied)


def listing(
    fragments: Sequence[BKB], cells: Cells
) -> tuple[tuple[Setting, tuple[StatementGap, ...]], ...]:
    """A dataset's gaps at every setting, in `SETTINGS` order.

    Args:
        fragments: The ontology fragments over the concepts the dataset's rows map.
        cells: The dataset, observed whole and per cell.
    """
    return tuple((setting, _gaps(setting, fragments, cells)) for setting in SETTINGS)


def log_sweep(swept: Sweep) -> None:
    """Log one `sweep ablation=` line per setting of `swept`."""
    for reading in swept.readings:
        logger.info(
            "sweep ablation=%s detector=%s grounding=%d credibility=%.2f sensitivity=%s "
            "specificity=%s",
            swept.ablation,
            reading.setting.definition,
            reading.setting.grounding,
            reading.setting.credibility,
            share(reading.detection.sensitivity),
            share(reading.detection.specificity),
        )


def log_choice(choice: Choice) -> None:
    """Log the `chosen` line: the level, or `none`, with the measures that chose it.

    `ablation=none` says the sweeps hold no `CHOOSING_ABLATION`; with it, `floor=n/a` says its
    count-based sensitivity has no eligible cell, and `level=none` with a floor says no level
    reaches it.
    """
    logger.info(
        "chosen level=%s ablation=%s floor=%s sensitivity=%s specificity=%s tied=%s",
        "none" if choice.level is None else f"{choice.level:.2f}",
        choice.ablation or "none",
        share(choice.floor),
        share(choice.sensitivity),
        share(choice.specificity),
        ",".join(f"{level:.2f}" for level in choice.tied) or "none",
    )


def log_listing(name: str, listed: Iterable[tuple[Setting, Sequence[StatementGap]]]) -> None:
    """Log one `sweep dataset=` line per setting: its gap and cell counts in dataset `name`."""
    for setting, found in listed:
        logger.info(
            "sweep dataset=%s detector=%s grounding=%d credibility=%.2f gaps=%d cells=%d",
            name,
            setting.definition,
            setting.grounding,
            setting.credibility,
            len(found),
            len({gap.cell for gap in found}),
        )


def _gaps(setting: Setting, fragments: Sequence[BKB], cells: Cells) -> tuple[StatementGap, ...]:
    with quiet_fusion():
        return gaps(
            setting.definition,
            fragments,
            cells,
            grounding=setting.grounding,
            credibility=setting.credibility,
        )
