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

from bko.baselines.request import Stratum
from bko.datasets.ablation import removed_count
from bko.evaluate import membership, perturb, perturb_threshold
from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import Dataset
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import (
    DIRECT_ARM,
    PERTURBED_ARM,
    PRAM_ARM,
    REAL_ARM,
    Experiment,
    Removal,
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
from bko.evaluate.perturb import (
    DELTA_EMPTY,
    DELTA_THIN,
    PERTURBING_ARMS,
    Delta,
    delta_for,
    refuse_unfixed,
)
from bko.evaluate.perturb_threshold import threshold
from bko.generate.flight import OVERHEAD_TOKENS
from bko.generate.messages import Settings
from bko.generate.spend import PRICES
from tests.test_evaluate_direct import SMALL_G2, g2_context
from tests.test_evaluate_launch import Starts
from tests.test_evaluate_shared import FORMAT


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


def test_the_pilots_deltas_are_not_yet_fixed():
    assert (DELTA_THIN, DELTA_EMPTY) == (None, None)
    assert delta_for(E10_PILOT_THIN) == Delta(None, "placeholder", 0.985)
    assert delta_for(E10_PILOT_EMPTY) == Delta(None, "placeholder", 1.0)
    assert Delta(None, "placeholder", 1.0).gate == pytest.approx(0.0)


def test_delta_for_reads_each_fraction_and_gates_any_other_run_at_zero(monkeypatch):
    monkeypatch.setattr(perturb, "DELTAS", {0.985: 0.011, 1.0: 0.022})
    assert delta_for(E10_PILOT_THIN) == Delta(0.011, "table", 0.985)
    assert delta_for(E10_PILOT_EMPTY) == Delta(0.022, "table", 1.0)
    assert delta_for(SMALL_G2) == Delta(0.0, "default", None)
    halved = dataclasses.replace(E10_PILOT_THIN, removal=Removal(("6-7", "male", 1), 0.5))
    assert delta_for(halved) == Delta(0.0, "default", 0.5)
    assert Delta(0.022, "table", 1.0).gate == pytest.approx(0.022)


def test_an_unfixed_delta_refuses_a_perturbing_run_but_a_rehearsal():
    for arm in sorted(PERTURBING_ARMS):
        alone = dataclasses.replace(
            E10_PILOT_THIN,
            arms=("none", DIRECT_ARM, arm),
            candidate=arm,
            comparators=("none", DIRECT_ARM),
        )
        with pytest.raises(EvaluationError, match="perturb-threshold"):
            refuse_unfixed(alone, run="x", rehearse=False)
        refuse_unfixed(alone, run="x", rehearse=True)
    unperturbed = dataclasses.replace(
        E10_PILOT_THIN, arms=("none", DIRECT_ARM), candidate=DIRECT_ARM, comparators=("none",)
    )
    refuse_unfixed(unperturbed, run="x", rehearse=False)


@pytest.mark.parametrize("name", ["e10-pilot-thin", "e10-pilot-empty"])
@pytest.mark.parametrize("command", ["score", "launch"])
def test_score_and_launch_refuse_a_pilot_at_its_unfixed_delta_before_any_file(
    tmp_path, caplog, name, command
):
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path / "runs", start=start)
    if command == "score":
        argv = ["score", name, "--log", str(tmp_path / "calls.jsonl")]
        argv += ["--candidates", str(tmp_path / "c.jsonl"), "--data-dir", str(tmp_path / "data")]
    else:
        argv = ["launch", "e10-1", name, "--dry-run"]
    with caplog.at_level(logging.ERROR):
        assert main(argv, context=context, out=io.StringIO()) == 2
    assert "perturb-threshold" in caplog.text
    assert sorted(path.name for path in tmp_path.iterdir()) == []
    assert start.commands == []


def test_a_pilot_rehearsal_launches_at_its_unfixed_delta(tmp_path):
    start = Starts()
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start)
    argv = ["launch", "e10-1", "e10-pilot-thin", "--rehearse", "--dry-run"]
    assert main(argv, context=context, out=io.StringIO()) == 0


def test_every_run_that_perturbs_is_on_the_pinned_diabetes_frame():
    perturbing = [name for name, run in RUNS.items() if PERTURBING_ARMS & set(run.experiment.arms)]
    assert perturbing == ["e10-pilot-thin", "e10-pilot-empty"]
    assert {RUNS[name].dataset for name in perturbing} == {Dataset.DIABETES}


def test_the_pilots_read_their_privacy_against_bko_direct():
    assert membership.REFERENCE["e10-pilot-thin"] == DIRECT_ARM
    assert membership.REFERENCE["e10-pilot-empty"] == DIRECT_ARM


def test_the_threshold_is_the_95th_percentile_of_the_steps():
    # Steps 0.02, 0.01 and 0.04 sort to 0.01, 0.02, 0.04; the 95th percentile sits 1.9 of the
    # way along their 2 gaps: 0.02 + 0.9 * (0.04 - 0.02) = 0.038.
    assert threshold([0.90, 0.92, 0.91, 0.95]) == pytest.approx(0.038)
    with pytest.raises(EvaluationError, match="no step"):
        threshold([0.9])
    for shapes in ([0.9, math.nan], [math.inf, 0.9, 0.91]):
        with pytest.raises(EvaluationError, match="not finite"):
            threshold(shapes)


THREE: Final = dataclasses.replace(
    SMALL_G2,
    holdout_seeds=(0, 1, 2),
    arms=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Three holdouts of `bko-direct` and `none`, no replica: two steps between their shapes."""


def _store(tmp_path: Path, caplog, experiment: Experiment = THREE) -> tuple[Path, Context]:
    """A `score g2` folder of `experiment` under `tmp_path`, its `rows` store, and the context."""
    caplog.set_level(logging.INFO)
    ctx, _ = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(experiment)}, runs_dir=tmp_path)
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
    (record,) = [r for r in caplog.records if r.getMessage().startswith("perturb threshold run=")]
    assert record.levelno == logging.WARNING
    found = re.fullmatch(
        r"perturb threshold run=g2 fraction=none holdouts=3 steps=2 percentile=95"
        r" delta=(\d\.\d{6})",
        record.getMessage(),
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


def test_perturb_threshold_needs_its_run_named(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        main(["perturb-threshold", "--rows", str(tmp_path / "rows.jsonl")], out=io.StringIO())
    assert raised.value.code == 2
    assert "run" in capsys.readouterr().err


def test_perturb_threshold_refuses_a_split_its_data_does_not_prepare(tmp_path, caplog, monkeypatch):
    store, ctx = _store(tmp_path, caplog)
    monkeypatch.setattr(perturb_threshold, "split_line", lambda _: "split other")
    argv = ["perturb-threshold", "g2", "--rows", str(store), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 3
    assert "holdout 0's split 'split other' is not" in caplog.text


def test_perturb_threshold_names_the_store_whose_rows_do_not_read(tmp_path, caplog, monkeypatch):
    store, ctx = _store(tmp_path, caplog)

    def unread(*_args: object, **_kwargs: object) -> pd.DataFrame:
        msg = "the rows do not hash to the arm line"
        raise IdentityError(msg)

    monkeypatch.setattr(perturb_threshold, "frame_of", unread)
    argv = ["perturb-threshold", "g2", "--rows", str(store), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 3
    assert f"the store {store}: the rows do not hash to the arm line" in caplog.text
