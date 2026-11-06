"""The perturbation pilot's named runs, their ceilings, `perturb-threshold` and their reference."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.datasets.ablation import removed_count
from bko.evaluate import membership
from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import Dataset
from bko.evaluate.errors import EvaluationError
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
from bko.evaluate.perturb import DELTA_EMPTY, DELTA_THIN, delta_for
from bko.evaluate.perturb_threshold import threshold
from bko.generate.flight import OVERHEAD_TOKENS
from bko.generate.messages import Settings
from bko.generate.spend import PRICES
from tests.test_evaluate_direct import SMALL_G2, g2_context
from tests.test_evaluate_shared import FORMAT


INPUT_PER_ROW: Final = 800
"""Input tokens a rewrite call is estimated to bill per row: E6's 200.8 input tokens per record
asked (`docs/EVIDENCE.md`, E6 Configuration), rounded up past the row the call shows, which is at
most a record, E6's 590.6 output tokens."""

OUTPUT_PER_ROW: Final = 600
"""Output tokens a rewrite call is estimated to bill per row: E6's 590.6 per record asked,
rounded up, for a reply that rewrites 19 of a record's fields."""


def asked(experiment: Experiment) -> int:
    """The rows a pilot holdout's gap fill asks for: its fill target less the cell's kept rows."""
    removal = experiment.removal
    assert removal is not None
    kept = experiment.fill_to - removed_count(removal.fraction, experiment.fill_to)
    return experiment.fill_to - kept


def estimate(experiment: Experiment, settings: Settings) -> float:
    """A pilot's rewrite spend in USD: every holdout's rows at `INPUT_PER_ROW`, `OUTPUT_PER_ROW`
    and `OVERHEAD_TOKENS` per call, input at the higher of `input` and `cache_write`."""
    rows = asked(experiment) * len(experiment.holdout_seeds)
    calls = math.ceil(asked(experiment) / settings.per_call) * len(experiment.holdout_seeds)
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


@pytest.mark.parametrize("pilot", [E10_PILOT_THIN, E10_PILOT_EMPTY], ids=["thin", "empty"])
def test_each_pilots_ceiling_covers_its_estimate(pilot):
    assert asked(pilot) == pilot.fill_to - (
        pilot.fill_to - removed_count(pilot.removal.fraction, pilot.fill_to)
    )
    assert 0 < estimate(pilot, E6_SETTINGS) <= pilot.spend_ceiling


def test_the_thin_pilot_asks_its_cells_removed_rows_and_the_empty_one_all_of_them():
    assert (asked(E10_PILOT_THIN), asked(E10_PILOT_EMPTY)) == (1250, 1269)


def test_the_pilots_deltas_are_not_yet_fixed():
    assert (DELTA_THIN, DELTA_EMPTY) == (0.0, 0.0)
    assert (delta_for(E10_PILOT_THIN), delta_for(E10_PILOT_EMPTY)) == (DELTA_THIN, DELTA_EMPTY)


def test_the_pilots_read_their_privacy_against_bko_direct():
    assert membership.REFERENCE["e10-pilot-thin"] == DIRECT_ARM
    assert membership.REFERENCE["e10-pilot-empty"] == DIRECT_ARM


def test_the_threshold_is_the_95th_percentile_of_the_steps():
    # Steps 0.02, 0.01 and 0.04 sort to 0.01, 0.02, 0.04; the 95th percentile sits 1.9 of the
    # way along their 2 gaps: 0.02 + 0.9 * (0.04 - 0.02) = 0.038.
    assert threshold([0.90, 0.92, 0.91, 0.95]) == pytest.approx(0.038)
    with pytest.raises(EvaluationError, match="no step"):
        threshold([0.9])


THREE: Final = dataclasses.replace(
    SMALL_G2,
    holdout_seeds=(0, 1, 2),
    arms=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Three holdouts of `bko-direct` and `none`, no replica: two steps between their shapes."""


def _store(tmp_path: Path, caplog) -> tuple[Path, Context]:
    """A `score g2` folder of `THREE` under `tmp_path`, its `rows` store, and the context."""
    caplog.set_level(logging.INFO)
    ctx, _ = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(THREE)}, runs_dir=tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    data = ["--data-dir", str(tmp_path / "data")]
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl"), *data]
    out = io.StringIO()
    caplog.clear()
    assert main(argv, context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    store = tmp_path / "rows.jsonl"
    argv = ["rows", "g2", "--folder", str(folder), "--rows", str(store), *data]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    caplog.clear()
    return store, ctx


def test_perturb_threshold_on_a_toy_store_logs_the_hand_computed_percentile(tmp_path, caplog):
    store, ctx = _store(tmp_path, caplog)
    argv = ["perturb-threshold", "g2", "--rows", str(store), "--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    table = pd.read_csv(io.StringIO(out.getvalue()))
    assert list(table.columns) == ["holdout", "shapes", "pair_trends", "step"]
    assert table["holdout"].tolist() == [0, 1, 2]
    shapes = table["shapes"].tolist()
    steps = sorted([abs(shapes[1] - shapes[0]), abs(shapes[2] - shapes[1])])
    assert table["step"].iloc[1:].tolist() == pytest.approx(
        [abs(shapes[1] - shapes[0]), abs(shapes[2] - shapes[1])], abs=1e-6
    )
    assert math.isnan(table["step"].iloc[0])
    # Two steps: the 95th percentile lies 0.95 of the way from the smaller to the larger.
    hand = steps[0] + Fraction(95, 100) * (steps[1] - steps[0])
    (line,) = [m for m in caplog.messages if m.startswith("perturb threshold run=")]
    found = re.fullmatch(
        r"perturb threshold run=g2 holdouts=3 steps=2 percentile=95 delta=(\d\.\d{6})", line
    )
    assert found is not None
    assert float(found[1]) == pytest.approx(float(hand), abs=2e-6)


def test_perturb_threshold_refuses_a_run_without_bko_direct(tmp_path, caplog):
    store, ctx = _store(tmp_path, caplog)
    lacking = dataclasses.replace(THREE, arms=("none", PRAM_ARM), candidate=PRAM_ARM)
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(lacking)})
    argv = ["perturb-threshold", "g2", "--rows", str(store), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"has no {DIRECT_ARM} arm" in caplog.text
