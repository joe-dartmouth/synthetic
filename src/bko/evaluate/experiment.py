"""E1's configuration, fixed before any run, its named runs, and the requests it gives every arm.

ADR `research/e1-evaluation-protocol` records the protocol and `docs/EVIDENCE.md` (E1) states how
each criterion is read. `RUNS` also holds G3's guidance-level runs, `g3-<level>`, one per level,
`g3-summary` being E1's level (`docs/EVIDENCE.md`, G3), the effort sweep's runs,
`g4-low` and `g4-medium` (G4), `g2`, the direct arm's run (`G2`), and `g5`, the prescribing arm's
run (`G5`), and E3's runs over the Synthea frame, `e3` and its holdout slices `e3-a` and `e3-b`
(ADR `research/synthea-downstream-design`). Each named run also fixes the dataset it scores.
No command-line flag changes a value here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.request import SEED_LIMIT, Method, Request, Strata, Stratum
from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import Dataset
from bko.evaluate.errors import EvaluationError
from bko.gaps.definitions import Definition
from bko.generate.messages import LEVELS, SETTINGS, Level, Settings


CRITERION_NAMES: Final = ("primary", "secondary", "privacy", "validity")
"""The criteria `compare` can judge, by name; `docs/EVIDENCE.md` (E1) states how each is read."""


@dataclass(frozen=True, slots=True)
class Removal:
    """A share of one cell's training rows that leaves every holdout's training split.

    Attributes:
        cell: The cell's age band, sex label and label.
        fraction: The share of its training rows removed, drawn per holdout at the holdout's seed
            (`bko.datasets.ablation.thinned`).
    """

    cell: Stratum
    fraction: float

    def __post_init__(self) -> None:
        """Refuse a fraction outside (0, 1).

        Raises:
            ValueError: If the fraction is not strictly between 0 and 1.
        """
        if not 0 < self.fraction < 1:
            msg = f"removal fraction {self.fraction} must lie strictly between 0 and 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Experiment:
    """What a run of the harness fixes before it starts.

    Attributes:
        holdout_seeds: One cell-stratified split per seed (`bko.datasets.split`); every request on
            a split takes its seed.
        arms: The arms scored on every split, by name.
        candidate: The arm the criteria judge.
        comparators: The arms the candidate must beat.
        replica_arms: The row-adding arms also asked for a replica of the training split's cells.
        fill_to: Each cell holding fewer training records than this is filled up to it.
        margin: What the candidate must beat each comparator by, in the measure's own units.
        privacy_margin: The share of the test rows' 5th-percentile distance the added rows' must
            reach.
        confidence: The two-sided level of every interval; not met reads each of a criterion's k
            intervals at 1 - (1 - confidence)/k.
        attempts: Each arm's tries at one request; a retry follows a refusal and takes a seed
            derived from the request's seed and the attempt.
        budget: The records a BKO arm's attempt may ask for, as a multiple of the request's rows.
        validity: The share of generated candidates that must pass every check, read as a point.
        spend_ceiling: The call log's spend in USD at which a run stops.
        criteria: The criteria `compare` judges, of `CRITERION_NAMES`; none reports every measure
            with no verdict.
        projection_after: The holdouts after which a paid run's spend, scaled to every holdout, is
            read against the ceiling (`bko.evaluate.watch`).
        idle_limit: The seconds a paid run's call may go without a reply, its retries included,
            before the run stops.
        removal: The training rows each holdout loses before any arm runs; None removes none.
        decontaminate: Whether the test records an arm copies leave the test split (E1's rule);
            E3 keeps them and counts them.
    """

    holdout_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    candidate: str
    comparators: tuple[str, ...]
    replica_arms: tuple[str, ...]
    fill_to: int = UNDERREPRESENTED_BELOW
    margin: float = 0.0
    privacy_margin: float = 0.9
    confidence: float = 0.95
    attempts: int = 5
    budget: int = 3
    validity: Fraction = Fraction(19, 20)
    spend_ceiling: float = 25.0
    criteria: tuple[str, ...] = CRITERION_NAMES
    projection_after: int = 5
    idle_limit: float = 900.0
    removal: Removal | None = None
    decontaminate: bool = True

    def __post_init__(self) -> None:
        """Refuse a configuration the harness cannot run or judge.

        Raises:
            ValueError: If there are fewer than 2 holdout seeds, a seed repeats or leaves
                `[0, SEED_LIMIT)`, an arm is named twice, a comparator or replica arm is not an
                arm, a replica arm adds no rows, the candidate is not an arm or is among its
                comparators, the confidence is not in (0, 1), there is no attempt, the budget is
                below 1, the validity is not in (0, 1], the spend ceiling is not above 0 USD, a
                criterion is unknown or named twice, the projection is not read after at least one
                holdout, or the idle limit is not above 0 seconds.
        """
        seeds = self.holdout_seeds
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            msg = f"holdout seeds {seeds} must be 2 or more distinct seeds"
            raise ValueError(msg)
        if not all(0 <= seed < SEED_LIMIT for seed in seeds):
            msg = f"every holdout seed must lie in [0, {SEED_LIMIT})"
            raise ValueError(msg)
        if len(set(self.arms)) != len(self.arms):
            msg = f"arms {self.arms} name an arm twice"
            raise ValueError(msg)
        stray = sorted((set(self.comparators) | set(self.replica_arms)) - set(self.arms))
        if stray:
            msg = f"{stray} are not arms of this experiment"
            raise ValueError(msg)
        controls = sorted(
            name for name in self.replica_arms if name in METHOD_ARMS and not Method(name).adds_rows
        )
        if controls:
            msg = f"{controls} add no rows, so they have no replica to fit"
            raise ValueError(msg)
        if self.candidate not in self.arms:
            msg = f"the candidate {self.candidate} is not an arm of this experiment"
            raise ValueError(msg)
        if self.candidate in self.comparators:
            msg = f"the candidate {self.candidate} cannot be one of its comparators"
            raise ValueError(msg)
        if not 0 < self.confidence < 1:
            msg = f"confidence {self.confidence} must lie strictly between 0 and 1"
            raise ValueError(msg)
        if self.attempts < 1:
            msg = f"attempts {self.attempts}: an arm needs at least one attempt"
            raise ValueError(msg)
        if self.budget < 1:
            msg = f"budget {self.budget}: an attempt must be able to ask for its request's rows"
            raise ValueError(msg)
        if not 0 < self.validity <= 1:
            msg = f"validity {self.validity} must lie in (0, 1]"
            raise ValueError(msg)
        if not self.spend_ceiling > 0:
            msg = f"spend ceiling {self.spend_ceiling} must be above 0 USD"
            raise ValueError(msg)
        if len(set(self.criteria)) != len(self.criteria) or not set(self.criteria) <= set(
            CRITERION_NAMES
        ):
            msg = f"criteria {self.criteria} must be distinct names among {CRITERION_NAMES}"
            raise ValueError(msg)
        self._check_stop_rules()

    def _check_stop_rules(self) -> None:
        if self.projection_after < 1:
            msg = f"projection after {self.projection_after} holdouts: read it after one at least"
            raise ValueError(msg)
        if not self.idle_limit > 0:
            msg = f"idle limit {self.idle_limit} must be above 0 seconds"
            raise ValueError(msg)


HOLDOUTS: Final = 100
"""E1's holdout count; its seeds are 0 to 99."""

METHOD_ARMS: Final = tuple(method.value for method in Method)
"""Every comparison method of `bko.baselines`, by name."""

ADDS_ROWS: Final = tuple(method.value for method in Method if method.adds_rows)
"""The comparison methods that add rows, by name: each also fills a replica."""

BKO_ARMS: Final[Mapping[str, Definition]] = MappingProxyType(
    {
        "bko": Definition.SUBGROUP,
        "bko-population": Definition.POPULATION,
        "bko-frequency": Definition.FREQUENCY,
    }
)
"""The BKO arms, each with the gap definition that guides it; `bko` is E1's candidate."""

DIRECT_ARM: Final = "bko-direct"
"""The free arm that draws records from the fused BKB with no API call (`bko.evaluate.direct`)."""

SKELETON_ARM: Final = "bko-skeleton"
"""The prescribing arm: a Messages arm guided by per-subgroup gaps, as `bko` is, whose every
request carries each record's concepts drawn from the fused BKB (`bko.evaluate.skeleton`). It is
not in `BKO_ARMS`, the three gap definitions' arms."""

MESSAGES_ARMS: Final[Mapping[str, Definition]] = MappingProxyType(
    {**BKO_ARMS, SKELETON_ARM: Definition.SUBGROUP}
)
"""Every arm that calls the Messages API, each with the gap definition that guides it."""

E1: Final = Experiment(
    holdout_seeds=tuple(range(HOLDOUTS)),
    arms=(*METHOD_ARMS, *BKO_ARMS),
    candidate="bko",
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *BKO_ARMS),
)
"""E1's configuration; `docs/EVIDENCE.md` (E1) names it."""

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
    arms=(*METHOD_ARMS, *BKO_ARMS),
    replica_arms=(*ADDS_ROWS, *BKO_ARMS),
    spend_ceiling=65.0,
    criteria=(),
    removal=Removal((">=65", "female", 1), 0.9),
    decontaminate=False,
)
"""E1's arms over the Synthea frame with 0.9 of `(>=65, female, 1)`'s training rows removed, test
copies kept, a $65.00 ceiling and no criterion (ADR `research/synthea-downstream-design`)."""

E3_SLICE_CEILING: Final = 32.5
"""Each of E3's two holdout slices' ceiling: half of E3's."""


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
            replace(E3, holdout_seeds=tuple(range(50)), spend_ceiling=E3_SLICE_CEILING),
            SETTINGS,
            Dataset.SYNTHEA,
        ),
        "e3-b": NamedRun(
            replace(E3, holdout_seeds=tuple(range(50, HOLDOUTS)), spend_ceiling=E3_SLICE_CEILING),
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


def flagged(
    train: pd.DataFrame, experiment: Experiment, *, edges: Sequence[int]
) -> tuple[Stratum, ...]:
    """The cells under `edges` holding fewer than `experiment.fill_to` records, in grid order."""
    return tuple(_short(train, experiment, edges=edges))


def gap_request(
    train: pd.DataFrame, experiment: Experiment, *, seed: int, frame: FrameSpec = UCI_FRAME
) -> Request:
    """Ask for every flagged cell of `frame` to be filled up to `experiment.fill_to` records.

    Raises:
        EvaluationError: If no cell is flagged, so the row-adding arms have nothing to fill.
    """
    quota = _short(train, experiment, edges=frame.edges)
    if not quota:
        msg = f"no cell holds fewer than {experiment.fill_to} training records; nothing to fill"
        raise EvaluationError(msg)
    return Request(quota=quota, strata=Strata.CELL, seed=seed, frame=frame)


def replica_request(train: pd.DataFrame, *, seed: int, frame: FrameSpec = UCI_FRAME) -> Request:
    """Ask for as many rows per cell as the training split holds: a synthetic copy of its cells."""
    quota = {cell: count for cell, count, _ in _cells(train, edges=frame.edges) if count}
    return Request(quota=quota, strata=Strata.CELL, seed=seed, frame=frame)


def _short(
    train: pd.DataFrame, experiment: Experiment, *, edges: Sequence[int]
) -> dict[Stratum, int]:
    return {
        cell: experiment.fill_to - count
        for cell, count, short in _cells(train, experiment.fill_to, edges=edges)
        if short
    }


def _cells(
    train: pd.DataFrame,
    below: int = UNDERREPRESENTED_BELOW,
    *,
    edges: Sequence[int],
) -> list[tuple[Stratum, int, bool]]:
    counts = cell_counts(train, edges=edges, below=below)
    return [
        ((str(band), str(sex), int(label)), int(count), bool(short))
        for band, sex, label, count, short in counts.itertuples(index=False)
    ]
