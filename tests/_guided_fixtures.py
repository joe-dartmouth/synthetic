"""A BKO arm over the ladder releases and the `UCI_X` archive, with a scripted send."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

import pandas as pd

from bko.baselines.request import Request, Strata
from bko.evaluate.guided import GuidedArm, Knowledge, spendable
from bko.gaps.definitions import Definition
from bko.gaps.scope import load_scope
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend
from tests._fusion_fixtures import UCI_X, training_split
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._ontology_fixtures import RoutingOpener


YOUNG_SICK: Final = ("<55", "female", 1)
YOUNG_WELL: Final = ("<55", "female", 0)
REQUEST: Final = Request(quota={YOUNG_SICK: 3}, strata=Strata.CELL, seed=0)
"""Three rows of young women with the condition."""


def knowledge(directory: Path) -> Knowledge:
    """The ladder releases loaded into `directory`, with `ANGINA` and `STENOSIS` as field rows."""
    pins, served = ladder_releases()
    rows = (ANGINA, STENOSIS)
    ontologies, fragments = load_scope(
        {row.concept for row in rows}, directory, releases=pins, open_url=RoutingOpener(served)
    )
    return Knowledge("uci", ontologies, fragments, rows)


def train(directory: Path) -> pd.DataFrame:
    """`UCI_X`'s default training split: ages 40 and 63, every measured value the first record's."""
    return training_split(directory, UCI_X)


def guided(
    directory: Path,
    send: AnySend,
    *,
    budget: int = 3,
    ceiling: float = 50.0,
    definition: Definition = Definition.SUBGROUP,
    idle: float | None = None,
) -> tuple[GuidedArm, CallLog]:
    """A BKO arm and its call log under `directory`; use the log in a `with` block.

    The arm sends through a `Flight` of one call at a time over `send`, under `ceiling` and the
    idle limit `idle`, None for none.
    """
    log = CallLog(directory / "calls.jsonl", SETTINGS)
    limits = Limits(1, idle)
    flight = Flight(send, log, limits, spendable=partial(spendable, log, ceiling), ceiling=ceiling)
    arm = GuidedArm(
        "bko", definition, knowledge(directory / "data"), log=log, budget=budget, flight=flight
    )
    return arm, log
