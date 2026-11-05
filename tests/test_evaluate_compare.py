"""`compare` over a `panel` scores file: each model's means and contrasts, the rest unchanged."""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from pathlib import Path

import numpy as np
import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.classifier import PANEL, Model
from bko.evaluate.context import Context
from bko.evaluate.experiment import Experiment
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from tests._evaluate_fixtures import SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARMS = ("none", "class-weight", "smote")

UNJUDGED = dataclasses.replace(
    SMALL,
    arms=ARMS,
    candidate="smote",
    comparators=("none", "class-weight"),
    replica_arms=(),
    criteria=(),
)
"""A run that judges no criterion, as `g2` and E6's runs do."""

JUDGED = dataclasses.replace(UNJUDGED, criteria=("primary", "secondary"))
"""An E1-like run that judges the separation criteria."""

BASES = ("recall", "precision", "auc-pr", "brier", "calibration-in-the-large")

PANEL_ONLY = re.compile(r"-(rf|et|mlp|nb)\b|brier|calibration-in-the-large|model=(rf|et|mlp|nb)")
"""What only a line of a panel model or a calibration measure holds."""


def scores(*, panel: bool) -> list[Score]:
    """Two holdouts of `ARMS`' augmented rows: every model's with `panel`, else LR's and HGB's.

    Both draw the same values, so the rows they share are equal.
    """
    rng = np.random.default_rng(0)
    found: list[Score] = []
    for holdout in SMALL.holdout_seeds:
        held = {Measure.HELD_OUT_RATIO: 0.25, Measure.DCR_P5: 0.02 + holdout / 100}
        found += [
            Score(holdout, HELD_OUT_ARM, Source.HELD_OUT, m, Scope.ALL, v) for m, v in held.items()
        ]
        for arm in ARMS:
            value = float(rng.uniform(0.01, 0.05))
            found.append(Score(holdout, arm, Source.ADDED, Measure.DCR_P5, Scope.ALL, value))
            for model in PANEL:
                suffix = "" if model is Model.LR else f"-{model}"
                for base in BASES:
                    for scope in (Scope.ALL, Scope.FLAGGED):
                        measure = Measure(f"{base}{suffix}")
                        value = float(rng.uniform(0.1, 0.9))
                        found.append(Score(holdout, arm, Source.AUGMENTED, measure, scope, value))
    return [s for s in found if panel or not PANEL_ONLY.search(s.measure.value)]


def write(tmp_path: Path, name: str, rows: list[Score]) -> Path:
    path = tmp_path / name
    path.write_text(render(rows), encoding="utf-8")
    return path


def compared(path: Path, experiment: Experiment, caplog) -> list[str]:
    """`compare`'s log lines over `path`, after its `compare` header."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        out = io.StringIO()
        assert main(["compare", str(path)], context=Context(experiment=experiment), out=out) == 0
    return [r.getMessage() for r in caplog.records if not r.getMessage().startswith("compare ")]


def test_compare_logs_each_panel_models_means_and_unjudged_contrasts(tmp_path, caplog):
    lines = compared(write(tmp_path, "panel.csv", scores(panel=True)), UNJUDGED, caplog)
    for model in PANEL:
        suffix = "" if model is Model.LR else f"-{model}"
        for base in ("recall", "auc-pr", "brier", "calibration-in-the-large"):
            assert any(
                line.startswith(f"mean arm=smote measure={base}{suffix} scope=all value=")
                for line in lines
            ), (model, base)
    for model in (Model.HGB, Model.RF, Model.ET, Model.MLP, Model.NB):
        found = [line for line in lines if line.startswith(f"unjudged model={model} ")]
        assert len(found) == 2 * 2 * len(UNJUDGED.comparators)
        assert all(
            re.fullmatch(
                rf"unjudged model={model} criterion=(primary|secondary) scope=\S+ arm=smote"
                rf" against=(none|class-weight) measure=(recall|auc-pr)-{model} mean=-?\d+\.\d{{6}}"
                r" low=\S+ high=\S+ refuting_high=\S+ holdouts=2 differing=2",
                line,
            )
            for line in found
        )
    assert not [line for line in lines if line.startswith("unjudged model=lr ")]


def test_a_judged_run_logs_each_panel_models_contrasts(tmp_path, caplog):
    lines = compared(write(tmp_path, "panel.csv", scores(panel=True)), JUDGED, caplog)
    for model in (Model.HGB, Model.RF, Model.ET, Model.MLP, Model.NB):
        found = [line for line in lines if line.startswith(f"contrast model={model} ")]
        assert len(found) == 2 * 2 * len(JUDGED.comparators), model
        assert all(
            f" measure=recall-{model} " in x or f" measure=auc-pr-{model} " in x for x in found
        )
    assert not [line for line in lines if line.startswith("unjudged")]


@pytest.mark.parametrize("experiment", [UNJUDGED, JUDGED], ids=["unjudged", "judged"])
def test_the_regression_and_boosting_lines_are_byte_unchanged(tmp_path, caplog, experiment):
    full = scores(panel=True)
    alone = [s for s in full if not PANEL_ONLY.search(s.measure.value)]
    assert alone == scores(panel=False)
    with_panel = compared(write(tmp_path, "panel.csv", full), experiment, caplog)
    without = compared(write(tmp_path, "default.csv", alone), experiment, caplog)
    assert [line for line in with_panel if not PANEL_ONLY.search(line)] == without
