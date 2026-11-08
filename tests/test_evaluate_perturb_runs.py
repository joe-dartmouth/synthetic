"""The perturbation pilot's named runs, their ceilings and their reference."""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import pytest

from bko.baselines.request import Stratum
from bko.datasets.ablation import removed_count
from bko.evaluate import membership
from bko.evaluate.data import Dataset
from bko.evaluate.experiment import (
    DIRECT_ARM,
    PERTURBED_ARM,
    PRAM_ARM,
    REAL_ARM,
    Experiment,
)
from bko.evaluate.named_runs import (
    E6_EMPTY,
    E6_SETTINGS,
    E6_THIN,
    E10_PILOT_EMPTY,
    E10_PILOT_THIN,
    RUNS,
    NamedRun,
)
from bko.generate.flight import OVERHEAD_TOKENS
from bko.generate.messages import Settings
from bko.generate.spend import PRICES


INPUT_PER_ROW: Final = 800
"""Input tokens a rewrite call is estimated to bill per row: E6's 200.8 input tokens per record
asked (`docs/EVIDENCE.md`, E6 Configuration), rounded up past the row the call shows, which is at
most a record, E6's 590.6 output tokens."""

OUTPUT_PER_ROW: Final = 600
"""Output tokens a rewrite call is estimated to bill per row: E6's 590.6 per record asked,
rounded up, for a reply that rewrites 19 of a record's fields."""


def quota(experiment: Experiment) -> dict[Stratum, int]:
    """A pilot holdout's gap request: every cell under the threshold, filled to `fill_to`.

    On Diabetes 130 the removed cell is the only one under E6's threshold on every holdout seed
    (`tests/test_datasets_diabetes_pinned.py`, which reads the pinned archive under `--releases`),
    so the request asks for that cell's fill target less the training rows its removal keeps.
    """
    removal = experiment.removal
    assert removal is not None
    kept = experiment.fill_to - removed_count(removal.fraction, experiment.fill_to)
    assert kept < experiment.threshold
    return {removal.cell: experiment.fill_to - kept}


def estimate(experiment: Experiment, settings: Settings) -> float:
    """A pilot's rewrite spend in USD: every requested cell's rows on every holdout at
    `INPUT_PER_ROW`, `OUTPUT_PER_ROW` and `OVERHEAD_TOKENS` per call, each cell cut into calls of
    `per_call` rows, input at the higher of `input` and `cache_write`."""
    holdouts = len(experiment.holdout_seeds)
    cells = quota(experiment).values()
    rows = sum(cells) * holdouts
    calls = sum(math.ceil(count / settings.per_call) for count in cells) * holdouts
    price = PRICES[settings.model]
    tokens_in = rows * INPUT_PER_ROW + calls * OVERHEAD_TOKENS
    rate = max(price.input, price.cache_write)
    return (tokens_in * rate + rows * OUTPUT_PER_ROW * price.output) / 1e6


@pytest.mark.parametrize(
    ("name", "pilot", "e6"),
    [
        ("e10-pilot-thin", E10_PILOT_THIN, E6_THIN),
        ("e10-pilot-empty", E10_PILOT_EMPTY, E6_EMPTY),
    ],
)
def test_runs_holds_each_pilot_with_its_arms(name, pilot, e6):
    assert RUNS[name] == NamedRun(pilot, E6_SETTINGS, Dataset.DIABETES)
    assert pilot == dataclasses.replace(
        e6,
        arms=("none", REAL_ARM, DIRECT_ARM, PERTURBED_ARM, PRAM_ARM),
        replica_arms=(),
        replica_rows=None,
        holdout_seeds=(0, 1),
        candidate=PERTURBED_ARM,
        comparators=(DIRECT_ARM, PRAM_ARM, "none"),
        spend_ceiling=5.0,
    )


@pytest.mark.parametrize(
    ("pilot", "usd"), [(E10_PILOT_THIN, 1.0315), (E10_PILOT_EMPTY, 1.0472)], ids=["thin", "empty"]
)
def test_each_pilots_ceiling_covers_its_estimate(pilot, usd):
    assert estimate(pilot, E6_SETTINGS) == pytest.approx(usd, abs=1e-4)
    assert estimate(pilot, E6_SETTINGS) <= pilot.spend_ceiling


def test_the_thin_pilot_asks_its_cells_removed_rows_and_the_empty_one_all_of_them():
    cell = ("6-7", "male", 1)
    assert (quota(E10_PILOT_THIN), quota(E10_PILOT_EMPTY)) == ({cell: 1250}, {cell: 1269})


def test_every_run_that_perturbs_is_on_the_pinned_diabetes_frame():
    perturbing = [
        name for name, run in RUNS.items() if {PERTURBED_ARM, PRAM_ARM} & set(run.experiment.arms)
    ]
    assert perturbing == ["e10-pilot-thin", "e10-pilot-empty"]
    assert {RUNS[name].dataset for name in perturbing} == {Dataset.DIABETES}


def test_the_pilots_read_their_privacy_against_bko_direct():
    assert membership.REFERENCE["e10-pilot-thin"] == DIRECT_ARM
    assert membership.REFERENCE["e10-pilot-empty"] == DIRECT_ARM
