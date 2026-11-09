"""`compare` over a `panel` scores file: each model's means and contrasts, the rest unchanged."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import math
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
    return [
        r.getMessage()
        for r in caplog.records
        if not r.getMessage().startswith(("compare ", "versions "))
    ]


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
    shared = [
        line
        for line in with_panel
        if not PANEL_ONLY.search(line) and not line.startswith("unjudged model=hgb ")
    ]
    assert shared == without


ORIGIN_LOGS = {
    "unjudged": (33, "d8bcbbfc348dc8828b9e94d16ef71c59df31ce7ea8eee8f297866f97315c668a"),
    "judged": (34, "ae223df7dc026828a15ac11142641e6aa866e3663f0dc10edca83799d9a42a8d"),
}
"""The line count and the sha256 of the `\\n`-joined lines that `compare` at c99e174, the commit
before the panel, logs after its header over `scores(panel=False)`, by experiment."""

ORIGIN_UNJUDGED = [
    (
        "unjudged criterion=primary scope=all arm=smote against=none measure=recall mean=0.043995"
        " low=-2.912156 high=3.000145 refuting_high=11.891448 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=primary scope=all arm=smote against=class-weight measure=recall"
        " mean=0.160316 low=-3.214536 high=3.535168 refuting_high=13.685811 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=primary scope=all arm=smote against=none measure=auc-pr mean=-0.430906"
        " low=-2.276585 high=1.414773 refuting_high=6.966077 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=primary scope=all arm=smote against=class-weight measure=auc-pr"
        " mean=-0.255045 low=-4.552115 high=4.042025 refuting_high=16.966450 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=secondary scope=flagged-subgroups arm=smote against=none measure=recall"
        " mean=0.345633 low=-2.320425 high=3.011691 refuting_high=11.030474 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=secondary scope=flagged-subgroups arm=smote against=class-weight"
        " measure=recall mean=-0.065338 low=-3.050668 high=2.919992 refuting_high=11.899058"
        " holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=secondary scope=flagged-subgroups arm=smote against=none measure=auc-pr"
        " mean=-0.267497 low=-2.106410 high=1.571417 refuting_high=7.102372 holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=secondary scope=flagged-subgroups arm=smote against=class-weight"
        " measure=auc-pr mean=-0.323079 low=-5.473547 high=4.827388 refuting_high=20.318604"
        " holdouts=2 differing=2"
    ),
    (
        "unjudged criterion=privacy scope=all arm=smote against=none measure=dcr-p5 mean=0.023340"
        " low=-0.090294 high=0.136975 refuting_high=0.136975 holdouts=2 differing=2"
    ),
]
"""The `unjudged` lines `compare` at c99e174 logs over `scores(panel=False)`, in order."""


@pytest.mark.parametrize("experiment", [UNJUDGED, JUDGED], ids=["unjudged", "judged"])
def test_a_default_files_compare_log_is_the_one_before_the_panel(tmp_path, caplog, experiment):
    lines = compared(write(tmp_path, "default.csv", scores(panel=False)), experiment, caplog)
    count, sha = ORIGIN_LOGS["judged" if experiment.criteria else "unjudged"]
    assert len(lines) == count
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == sha
    if not experiment.criteria:
        assert [line for line in lines if line.startswith("unjudged")] == ORIGIN_UNJUDGED


def partial(rows: list[Score], *, blank: bool) -> list[Score]:
    """`rows` less `class-weight`'s random forest rows; with `blank`, smote's flagged-subgroup
    random forest recall is NaN on holdout 0, which only an unjudged reading skips."""
    kept = [s for s in rows if not (s.arm == "class-weight" and s.measure.value.endswith("-rf"))]
    at = (0, "smote", Measure.RECALL_RF, Scope.FLAGGED)
    return [
        dataclasses.replace(s, value=math.nan)
        if blank and (s.holdout, s.arm, s.measure, s.scope) == at
        else s
        for s in kept
    ]


def test_an_arm_without_a_models_rows_is_skipped_in_its_unjudged_lines(tmp_path, caplog):
    lines = compared(
        write(tmp_path, "partial.csv", partial(scores(panel=True), blank=True)), UNJUDGED, caplog
    )
    found = [line for line in lines if line.startswith("unjudged model=rf ")]
    assert found[0] == "unjudged model=rf skip arm=smote against=class-weight reason=absent"
    assert (
        "unjudged model=rf skip criterion=secondary scope=flagged-subgroups arm=smote against=none"
        " measure=recall-rf reason=undefined defined=1 holdouts=2"
    ) in found
    rest = found[1:]
    assert len(rest) == 2 * 2
    assert all(" against=none " in line for line in rest)
    other = [line for line in lines if line.startswith("unjudged model=et ")]
    assert len(other) == 2 * 2 * len(UNJUDGED.comparators)


def test_an_arm_without_a_models_rows_is_skipped_in_its_contrasts(tmp_path, caplog):
    lines = compared(
        write(tmp_path, "partial.csv", partial(scores(panel=True), blank=False)), JUDGED, caplog
    )
    found = [line for line in lines if line.startswith("contrast ") and " model=rf " in line]
    assert found[0] == "contrast skip model=rf arm=smote against=class-weight reason=absent"
    assert len(found) == 1 + 2 * 2


def test_compares_log_opens_with_its_versions_line(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("bko.evaluate.provenance.commit", lambda: ("0123abc", False))
    path = write(tmp_path, "default.csv", scores(panel=False))
    with caplog.at_level(logging.INFO):
        out = io.StringIO()
        context = Context(experiment=UNJUDGED)
        assert main(["compare", str(path)], context=context, out=out) == 0
    assert caplog.messages[0].startswith("versions commit=")
