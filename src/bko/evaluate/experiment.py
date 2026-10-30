"""E1's configuration, fixed before any run, and the requests it gives every arm.

ADR `research/e1-evaluation-protocol` records the protocol and `docs/EVIDENCE.md` (E1) states how
each criterion is read. `bko.evaluate.named_runs` holds every named run built from it. No
command-line flag changes a value here.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.request import SEED_LIMIT, Method, Request, Strata, Stratum
from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError
from bko.gaps.definitions import Definition


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
                holdout, the idle limit is not above 0 seconds, or `REAL_ARM` is an arm with no
                removal, a replica arm, or an arm of an experiment that removes copies.
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
            name for name in self.replica_arms if name in Method and not Method(name).adds_rows
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
        self._check_real()

    def _check_real(self) -> None:
        """Refuse `REAL_ARM` where the harness cannot score it.

        Its added rows are the removal's rows, so it needs a removal; it has no runner, so it
        fills no replica; and its rows are real training rows, so E3's copy rule, which keeps
        copies, is the one it is read under.
        """
        if REAL_ARM in self.replica_arms:
            msg = f"the reference arm {REAL_ARM!r} has no runner, so it has no replica"
            raise ValueError(msg)
        if REAL_ARM not in self.arms:
            return
        if self.removal is None:
            msg = f"the reference arm {REAL_ARM!r} needs a removal: its rows are the removed rows"
            raise ValueError(msg)
        if self.decontaminate:
            msg = f"the reference arm {REAL_ARM!r} needs an experiment that keeps copies"
            raise ValueError(msg)

    def _check_stop_rules(self) -> None:
        if self.projection_after < 1:
            msg = f"projection after {self.projection_after} holdouts: read it after one at least"
            raise ValueError(msg)
        if not self.idle_limit > 0:
            msg = f"idle limit {self.idle_limit} must be above 0 seconds"
            raise ValueError(msg)


HOLDOUTS: Final = 100
"""E1's holdout count; its seeds are 0 to 99."""

METHOD_ARMS: Final[tuple[str, ...]] = tuple(
    map(
        str,
        (
            Method.NONE,
            Method.CLASS_WEIGHT,
            Method.SMOTE,
            Method.ADASYN,
            Method.CTGAN,
            Method.TVAE,
        ),
    )
)
"""docs/RESEARCH.md section 4's comparison set, by name. A method `bko.baselines` adds later joins
`bko.evaluate.run.ARMS` alone, so E1 and E3 keep these six."""

ADDS_ROWS: Final = tuple(name for name in METHOD_ARMS if Method(name).adds_rows)
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

REAL_ARM: Final = "real"
"""The reference arm: its added rows are the training rows the experiment's removal took, so its
augmented training set is the un-ablated training split. It has no runner and no `check`."""


class Curation(enum.StrEnum):
    """How a Messages arm curates its accepted rows before they count toward its quota."""

    CLASSIFIER = "classifier"
    LIKELIHOOD = "likelihood"


@dataclass(frozen=True, slots=True)
class MessagesArm:
    """How a Messages arm builds its requests and keeps its rows.

    Attributes:
        definition: The gap definition that guides it; None for guidance by cell alone, which
            reads no gap.
        prescribes: Whether each request carries every record's concepts drawn from the fused BKB
            (`bko.evaluate.skeleton`).
        shows: Whether each request shows training records.
        curation: How accepted rows are curated before they count; None keeps every one.
        over_ask: The multiple of a cell's shortfall a ranked curation asks for.
        budget: The records an attempt may ask for, as a multiple of the request's rows; None reads
            `Experiment.budget`.
    """

    definition: Definition | None
    prescribes: bool = False
    shows: bool = False
    curation: Curation | None = None
    over_ask: int = 1
    budget: int | None = None

    def __post_init__(self) -> None:
        """Refuse an inconsistent over-ask or a budget under 1.

        Raises:
            ValueError: If the spec is inconsistent.
        """
        if self.over_ask < 1 or (self.over_ask > 1) != (self.curation is Curation.LIKELIHOOD):
            msg = f"over_ask {self.over_ask} needs likelihood curation and only it"
            raise ValueError(msg)
        if self.budget is not None and self.budget < 1:
            msg = f"budget {self.budget} must be 1 or more"
            raise ValueError(msg)


CLLM_ARM: Final = "cllm"
"""The CLLM baseline (Seedat et al. 2024): each call shows training records of its cell's groups and
reads no gap, and a classifier fit on the training rows drops what it confidently puts at the other
label (`bko.evaluate.cllm`, `bko.evaluate.curation`)."""

CLLM_RAW_ARM: Final = "cllm-uncurated"
"""`cllm` with every accepted row kept: the same prompt, no curation."""

CLLM_BUDGET: Final = 3
"""The CLLM arms' attempt budget, `max(3, ceil(1.5 / s))` for the least curator keep share `s` the
pilot measured (ADR `research/cllm-arm-design`, The pilot)."""

MESSAGES_ARMS: Final[Mapping[str, MessagesArm]] = MappingProxyType(
    {
        **{name: MessagesArm(definition) for name, definition in BKO_ARMS.items()},
        SKELETON_ARM: MessagesArm(Definition.SUBGROUP, prescribes=True),
        CLLM_ARM: MessagesArm(None, shows=True, curation=Curation.CLASSIFIER, budget=CLLM_BUDGET),
        CLLM_RAW_ARM: MessagesArm(None, shows=True, budget=CLLM_BUDGET),
    }
)
"""Every arm that calls the Messages API, by name, with how it builds requests and keeps rows."""

E1: Final = Experiment(
    holdout_seeds=tuple(range(HOLDOUTS)),
    arms=(*METHOD_ARMS, *BKO_ARMS),
    candidate="bko",
    comparators=METHOD_ARMS,
    replica_arms=(*ADDS_ROWS, *BKO_ARMS),
)
"""E1's configuration; `docs/EVIDENCE.md` (E1) names it."""


def flagged(
    train: pd.DataFrame, experiment: Experiment, *, edges: Sequence[int]
) -> tuple[Stratum, ...]:
    """The cells under `edges` holding fewer than `experiment.fill_to` records, in grid order."""
    return tuple(_short(train, experiment, edges=edges))


def gap_request(
    train: pd.DataFrame, experiment: Experiment, *, seed: int, frame: FrameSpec
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


def replica_request(train: pd.DataFrame, *, seed: int, frame: FrameSpec) -> Request:
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
