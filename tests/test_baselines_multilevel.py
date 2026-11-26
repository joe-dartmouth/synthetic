"""The multilevel baseline `hb-marginals` and its per-column fit."""

from __future__ import annotations

import dataclasses
import functools
import logging

import numpy as np
import pandas as pd
import pytest

from bko.baselines import multilevel, multilevel_fit
from bko.baselines.categories import categorise
from bko.baselines.errors import BaselineError
from bko.baselines.multilevel import hb_marginals
from bko.baselines.multilevel_fit import (
    EVIDENCE_DECIMALS,
    GTOL,
    HALVINGS,
    MAIN_SCALE,
    MAX_PARAMETERS,
    MAXITER,
    NEWTON_STEPS,
    OPTIMISER,
    TAU_GRID,
    Fit,
    fit_column,
    fit_pair,
)
from bko.baselines.pooling import marginals
from bko.baselines.records import Baseline, check
from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.cells import LABEL, cell_keys
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import EDGE, full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FRAME = dataclasses.replace(UCI_FRAME, edges=(50, 60))
"""UCI's frame banded `<50`, `50-59` and `>=60`, as `tests/test_baselines_pooling.py` bands it."""

TRAIN = full_records()

QUOTA: dict[Stratum, int] = {
    ("<50", "female", 1): 3,
    ("50-59", "male", 1): 4,
    (">=60", "female", 0): 2,
}
"""Rows to add in three cells, one per band."""

GRID: tuple[Stratum, ...] = tuple(
    (band, sex, label) for band in ("a0", "a1") for sex in ("female", "male") for label in (0, 1)
)
"""A 2x2x2 grid of cells for the fit's own tests."""


@functools.cache
def fill(seed: int) -> Baseline:
    """`hb_marginals`' fill of `QUOTA` at `seed`, computed once; no test mutates it."""
    return hb_marginals(TRAIN, Request(quota=QUOTA, seed=seed, frame=FRAME))


def shares(rule: dict[Stratum, float], *, records: int = 20) -> dict[Stratum, dict[object, int]]:
    """Each cell's counts of `x` and `y`, `x` at the cell's share of `records`."""
    counts: dict[Stratum, dict[object, int]] = {}
    for cell, share in rule.items():
        held = round(share * records)
        counts[cell] = {"x": held, "y": records - held}
    return counts


def share_of_x(fit: Fit, cell: Stratum) -> float:
    return fit.predict(cell)[fit.categories.index("x")]


def by_sex(emptied: Stratum) -> dict[Stratum, dict[object, int]]:
    """`x` at 0.9 in every female cell and 0.1 in every male one, `emptied` holding no record."""
    counts = shares({cell: 0.9 if cell[1] == "female" else 0.1 for cell in GRID})
    counts[emptied] = {"x": 0, "y": 0}
    return counts


@pytest.mark.parametrize(
    ("emptied", "above"),
    [(("a0", "female", 1), True), (("a1", "male", 0), False)],
    ids=["female", "male"],
)
def test_an_empty_cell_is_predicted_from_its_margins(emptied, above):
    share = share_of_x(fit_column(by_sex(emptied)), emptied)
    assert (share > 0.5) is above


def sex_split(per_cell: int = 20) -> pd.DataFrame:
    """`full_records` with `exang` 1 in 90% of each female cell and 10% of each male one."""
    train = full_records(per_cell)
    cell = [train["age"] >= EDGE, "sex", LABEL]
    position = train.groupby(cell).cumcount()
    cut = np.where(train["sex"] == 0, 0.9 * per_cell, 0.1 * per_cell)
    return train.assign(exang=(position < cut).astype(train["exang"].dtype))


def test_hb_reads_the_sex_margin_where_banded_pooling_reads_the_other_sex():
    frame = dataclasses.replace(UCI_FRAME, edges=(EDGE,))
    train = sex_split()
    emptied = (train["age"] < EDGE) & (train["sex"] == 0) & (train[LABEL] == 1)
    train = train.loc[~emptied]
    request = Request(quota={(f"<{EDGE}", "female", 1): 200}, frame=frame)
    hb = hb_marginals(train, request).added["exang"].mean()
    pooled = marginals(train, request).added["exang"].mean()
    assert hb - pooled > 0.5


def test_a_two_way_term_is_learned_from_the_cells_that_identify_it():
    emptied = ("a0", "female", 1)
    rule = {cell: 0.9 if cell[0] == "a0" and cell[2] == 1 else 0.1 for cell in GRID}
    counts = shares(rule, records=40)
    counts[emptied] = {"x": 0, "y": 0}
    assert share_of_x(fit_column(counts), emptied) > 0.5


def test_a_thin_cell_is_pulled_toward_its_margins():
    thin = ("a0", "female", 1)
    counts = shares(dict.fromkeys(GRID, 0.1), records=50)
    counts[thin] = {"x": 3, "y": 0}
    assert 0.1 < share_of_x(fit_column(counts), thin) < 1.0


def test_the_scales_are_grid_values():
    fit = fit_column(by_sex(("a0", "female", 1)))
    assert fit.tau2 in TAU_GRID
    assert fit.tau3 in TAU_GRID
    assert fit.evidence == round(fit.evidence, EVIDENCE_DECIMALS)


def test_the_fit_is_deterministic():
    counts = by_sex(("a0", "female", 1))
    first, second = fit_column(counts), fit_column(counts)
    assert (first.tau2, first.tau3, first.evidence, first.iterations) == (
        second.tau2,
        second.tau3,
        second.evidence,
        second.iterations,
    )
    assert first.categories == second.categories
    for cell in GRID:
        assert first.predict(cell) == second.predict(cell)
    again = hb_marginals(TRAIN, Request(quota=QUOTA, seed=3, frame=FRAME))
    pd.testing.assert_frame_equal(fill(3).added, again.added)


@pytest.mark.parametrize(
    ("tau2", "tau3", "dropped"),
    [(0.0, 0.5, {"AS", "AY", "SY"}), (0.5, 0.0, {"E"}), (0.0, 0.0, {"AS", "AY", "SY", "E"})],
    ids=["two-way", "cell", "both"],
)
def test_a_zero_scale_removes_its_terms(tau2, tau3, dropped):
    emptied = ("a0", "female", 1)
    fit = fit_pair(by_sex(emptied), tau2, tau3)
    kinds = {key[0] for key in fit.index}
    assert not kinds & dropped
    assert kinds >= {"mu", "A", "S", "Y"}
    assert fit.coefficients.shape[0] == len(fit.index)
    for cell in GRID:
        assert sum(fit.predict(cell)) == pytest.approx(1.0)
    assert share_of_x(fit, emptied) > 0.5


def test_the_main_effects_alone_predict_every_cell_of_a_sex_from_its_margin():
    fit = fit_pair(by_sex(("a0", "female", 1)), 0.0, 0.0)
    female = [share_of_x(fit, cell) for cell in GRID if cell[1] == "female"]
    assert female == pytest.approx([female[0]] * len(female), abs=1e-3)


def test_fit_column_keeps_the_best_pair():
    counts = by_sex(("a0", "female", 1))
    best = fit_column(counts)
    pairs = [fit_pair(counts, tau2, tau3) for tau2 in TAU_GRID for tau3 in TAU_GRID]
    assert best.evidence == max(fit.evidence for fit in pairs)
    first = next(fit for fit in pairs if fit.evidence == best.evidence)
    assert (best.tau2, best.tau3) == (first.tau2, first.tau3)


def test_the_mode_reaches_the_gradient_tolerance():
    for name in ("chol", "oldpeak", "cp"):
        group = TRAIN.loc[TRAIN[LABEL] == 1, name]
        counts = _cell_counts(name, group)
        for tau2, tau3 in ((0.0, 0.0), (1 / 64, 1 / 32), (2.0, 2.0)):
            fit = fit_pair(counts, tau2, tau3)
            assert fit.gradient < GTOL, (name, tau2, tau3, fit.gradient)


def _cell_counts(name: str, group: pd.Series) -> dict[Stratum, dict[object, int]]:
    """`name`'s categories under the label-1 group, counted per cell of `FRAME`'s grid."""
    categories = categorise(TRAIN[name], group, ordered=name in FRAME.numeric)
    cells = cell_keys(TRAIN, edges=FRAME.edges)
    return {
        stratum: categories[
            (cells["age_band"] == stratum[0])
            & (cells["sex"] == stratum[1])
            & (cells[LABEL] == stratum[2])
        ]
        .value_counts()
        .to_dict()
        for stratum in Strata.CELL.grid(edges=FRAME.edges)
    }


def test_the_evidence_does_not_depend_on_the_start(monkeypatch):
    counts = _cell_counts("oldpeak", TRAIN.loc[TRAIN[LABEL] == 1, "oldpeak"])
    pairs = ((0.0, 0.0), (1 / 16, 1 / 64), (0.5, 0.25), (2.0, 2.0))
    zeros = [fit_pair(counts, tau2, tau3) for tau2, tau3 in pairs]
    original = multilevel_fit.minimize
    rng = np.random.default_rng(0)

    def perturbed(objective, start, **options):
        return original(objective, start + rng.normal(0.0, 0.5, start.size), **options)

    monkeypatch.setattr(multilevel_fit, "minimize", perturbed)
    moved = [fit_pair(counts, tau2, tau3) for tau2, tau3 in pairs]
    for first, second in zip(zeros, moved, strict=True):
        assert first.evidence == second.evidence
        assert np.allclose(first.coefficients, second.coefficients, atol=1e-6)


def test_a_fit_that_does_not_converge_is_refused(monkeypatch):
    monkeypatch.setattr(multilevel_fit, "MAXITER", 1)
    with pytest.raises(BaselineError, match="converge"):
        fit_column(by_sex(("a0", "female", 1)))


def test_a_column_with_too_many_parameters_is_refused(monkeypatch):
    counts = by_sex(("a0", "female", 1))
    monkeypatch.setattr(multilevel_fit, "MAX_PARAMETERS", 3)
    with pytest.raises(BaselineError, match="parameters, above MAX_PARAMETERS 3"):
        fit_pair(counts, 0.5, 0.5)


def test_an_optimiser_stop_short_of_maxiter_is_left_to_the_newton_stage(monkeypatch):
    counts = by_sex(("a0", "female", 1))
    expected = fit_pair(counts, 0.5, 0.5)
    original = multilevel_fit.minimize

    def abnormal(objective, start, **options):
        result = original(objective, start, **{**options, "options": {"maxiter": 3}})
        result.success, result.message = False, "ABNORMAL"
        return result

    monkeypatch.setattr(multilevel_fit, "minimize", abnormal)
    fit = fit_pair(counts, 0.5, 0.5)
    assert fit.gradient < GTOL
    assert np.allclose(fit.coefficients, expected.coefficients, atol=1e-6)


def test_a_fit_and_a_column_hold_read_only_arrays():
    fit = fit_column(by_sex(("a0", "female", 1)))
    assert not fit.coefficients.flags.writeable
    column = multilevel._column(
        TRAIN, cell_keys(TRAIN, edges=FRAME.edges), Request(quota=QUOTA, frame=FRAME), "chol", 1
    )
    for array in (column.values, column.categories, column.label_group):
        assert not array.flags.writeable


def test_probabilities_sum_to_one():
    counts = {cell: {"x": 4, "y": 9, "z": 2} for cell in GRID}
    counts["a1", "male", 0] = {}
    fit = fit_column(counts)
    assert len(fit.categories) == 3
    for cell in (*GRID, ("a2", "female", 1)):
        probabilities = fit.predict(cell)
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(p > 0 for p in probabilities)


def test_a_one_category_column_predicts_it_with_certainty():
    fit = fit_column({cell: {"x": 5} for cell in GRID})
    assert fit.categories == ("x",)
    assert fit.predict(GRID[0]) == (1.0,)


def test_every_added_row_passes_check():
    check(fill(3), TRAIN, Request(quota=QUOTA, seed=3, frame=FRAME))


def test_a_request_by_label_is_refused():
    request = Request(quota={(1,): 3}, strata=Strata.LABEL, frame=FRAME)
    with pytest.raises(BaselineError, match="keyed by cell"):
        hb_marginals(TRAIN, request)


def test_a_retry_seed_changes_the_fill():
    assert not fill(3).added.equals(fill(4).added)


def test_each_fit_logs_one_line_and_each_cell_one_pool_line(caplog):
    request = Request(quota=QUOTA, frame=FRAME)
    with caplog.at_level(logging.INFO, logger="bko.baselines"):
        hb_marginals(TRAIN, request)
    lines = [line for line in caplog.messages if line.startswith("hb ")]
    fitted = [name for name in FRAME.columns if name not in {"age", "sex", FRAME.label}]
    assert len(lines) == len({stratum[2] for stratum in QUOTA}) * len(fitted)
    assert lines[0].startswith(f"hb holdout=-1 fill=none label=1 column={fitted[0]} tau2=")
    assert all(" evidence=" in line and " iterations=" in line for line in lines)
    pools = [line for line in caplog.messages if line.startswith("pool ")]
    assert len(pools) == len(QUOTA)
    assert pools[0].startswith("pool holdout=-1 fill=none method=hb-marginals key=<50/female/1 ")


def test_every_constant_is_pinned():
    assert TAU_GRID == (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
    assert MAIN_SCALE == 10.0  # ruff: ignore[float-equality-comparison]
    assert OPTIMISER == "L-BFGS-B"
    assert GTOL == 1e-8  # ruff: ignore[float-equality-comparison]
    assert MAXITER == 1000
    assert EVIDENCE_DECIMALS == 6
    assert NEWTON_STEPS == 20
    assert HALVINGS == 30
    assert MAX_PARAMETERS == 4096
