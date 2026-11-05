"""`python -m bko.evaluate panel`: a folder's stored fills scored under the classifier panel."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from bko.baselines.errors import ShortfallError
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, number
from bko.baselines.request import Request
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.classifier import (
    MODELS,
    PANEL,
    Factory,
    Model,
    build,
    build_et,
    build_hgb,
    build_rf,
)
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment, Removal
from bko.evaluate.panel import StoredBaseline, admitted
from bko.evaluate.rows_read import read_rows
from bko.evaluate.run_folder import _extends, read_folder
from bko.evaluate.scores import Measure, Source
from tests._baselines_fixtures import cell_list
from tests._evaluate_bko_fixtures import NARROW
from tests._evaluate_fixtures import FAST_ARMS
from tests.test_evaluate_shared import rescorer, run_folder


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _small(factory: Factory) -> Factory:
    """`factory` at 5 trees, so a test's forests fit fast."""
    return lambda frame: factory(frame).set_params(model__n_estimators=5)


FAST_PANEL: Mapping[Model, Factory] = {
    **PANEL,
    Model.RF: _small(build_rf),
    Model.ET: _small(build_et),
}
"""`PANEL` in its order, with small forests."""

MIXED = dataclasses.replace(
    NARROW,
    arms=("none", "class-weight", "smote", "bko"),
    candidate="bko",
    comparators=("none", "class-weight", "smote"),
    criteria=(),
    replica_arms=("smote",),
    decontaminate=False,
)
"""A run of each kind of runner `panel` answers: regenerated controls, a stored baseline with a
replica, and a BKO arm. SMOTE's replica copies test records, so the run keeps them."""

MIXED_ARMS = {name: FAST_ARMS[name] for name in ("none", "class-weight", "smote")}


class Spy:
    """A runner that records each call and answers through `runner`."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.calls = 0

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        self.calls += 1
        return self.runner(train, request)


def folder_and_store(
    tmp_path: Path, caplog, experiment: Experiment, arms: Mapping[str, Runner]
) -> tuple[Path, Path]:
    """A run folder of `experiment` under `arms`, and the `rows` store of it; both exit 0."""
    folder = run_folder(tmp_path, "run", caplog, experiment=experiment, arms=arms)
    store = tmp_path / "run-rows" / "rows.jsonl"
    store.parent.mkdir()
    argv = ["rows", "--folder", str(folder), "--rows", str(store)]
    argv += ["--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=rescorer(experiment=experiment, arms=arms), out=out) == 0
    caplog.clear()
    return folder, store


def paneled(
    tmp_path: Path,
    folder: Path,
    store: Path,
    *,
    experiment: Experiment,
    arms: Mapping[str, Runner],
    models: Mapping[Model, Factory] = FAST_PANEL,
) -> tuple[int, str]:
    """Run `panel` over `folder` and `store`; return its exit code and stdout."""
    argv = ["panel", "--folder", str(folder), "--rows", str(store)]
    argv += ["--data-dir", str(tmp_path / "data")]
    ctx = dataclasses.replace(rescorer(experiment=experiment, arms=arms), panel=models)
    out = io.StringIO()
    code = main(argv, context=ctx, out=out)
    return code, out.getvalue()


def error(caplog) -> str:
    """The one ERROR line logged."""
    (found,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    return found


def less_panel(text: str) -> str:
    """A scores CSV without the rows `panel` admits beyond the folder's."""
    lines = text.splitlines(keepends=True)
    kept = [lines[0]]
    for line in lines[1:]:
        holdout, arm, source, measure, scope, _ = line.split(",")
        if not admitted((int(holdout), arm, source, measure, scope)):
            kept.append(line)
    return "".join(kept)


@pytest.mark.parametrize(
    ("folder", "current", "extends"),
    [
        ("lr,hgb", "lr,hgb,rf,et,mlp,nb", True),
        ("lr,hgb", "lr,hgb", True),
        ("lr,hgb", "lr,rf", False),
        ("lr,hgb", "hgb,lr,rf", False),
        ("lr,hgb", "lr", False),
        ("", "lr", False),
    ],
)
def test_a_current_models_list_extends_a_folders_only_from_its_end(folder, current, extends):
    assert _extends(folder, current) is extends


def test_panel_scores_every_model_and_check_2_accepts_its_models_list(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    with caplog.at_level(logging.INFO):
        code, text = paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS)
    assert code == 0
    assert " models=lr,hgb,rf,et,mlp,nb" in caplog.text
    augmented = {line.split(",")[3] for line in text.splitlines()[1:] if ",augmented," in line}
    for model in PANEL:
        suffix = "" if model is Model.LR else f"-{model}"
        expected = {f"{m}{suffix}" for m in ("recall", "precision", "auc-pr", "brier")}
        assert expected | {f"calibration-in-the-large{suffix}"} <= augmented


@pytest.mark.parametrize(
    "models",
    [
        {Model.LR: build, Model.RF: build_rf},
        {Model.HGB: build_hgb, Model.LR: build, Model.RF: build_rf},
        {Model.LR: build},
    ],
    ids=["lr,rf", "hgb,lr,rf", "lr"],
)
def test_check_2_refuses_a_models_list_the_folders_is_not_a_prefix_of(tmp_path, caplog, models):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    code = paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS, models=models)
    assert code == (3, "")
    names = ",".join(models)
    assert re.fullmatch(
        rf"{re.escape(str(folder))}'s models=lr,hgb is not a prefix of the run 'e1''s"
        rf" models={names}",
        error(caplog),
    )


def test_check_2_refuses_an_edited_fill_to(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    edited = dataclasses.replace(MIXED, fill_to=MIXED.fill_to + 1)
    assert paneled(tmp_path, folder, store, experiment=edited, arms=MIXED_ARMS) == (3, "")
    assert re.fullmatch(
        rf"the experiment lines of {re.escape(str(folder))} and the run 'e1' now differ:"
        rf" \{{'fill_to': \('{MIXED.fill_to}', '{MIXED.fill_to + 1}'\)\}}",
        error(caplog),
    )


def test_panel_answers_baselines_from_the_store(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    spies = {name: Spy(runner) for name, runner in MIXED_ARMS.items()}
    code, _ = paneled(tmp_path, folder, store, experiment=MIXED, arms=spies)
    assert code == 0
    assert spies["smote"].calls == 0
    holdouts = len(MIXED.holdout_seeds)
    assert spies["none"].calls == spies["class-weight"].calls == holdouts


def test_panel_output_less_panel_rows_is_the_folders_scores_csv(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    with caplog.at_level(logging.INFO):
        code, text = paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS)
    assert code == 0
    assert text != (folder / "scores.csv").read_text(encoding="utf-8")
    assert less_panel(text) == (folder / "scores.csv").read_text(encoding="utf-8")
    assert r"panel holdout=1 differing=0 subsampled=false" in caplog.text


def test_a_moved_regression_value_exits_3(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    moved = {**FAST_PANEL, Model.LR: lambda frame: build(frame).set_params(model__C=0.01)}
    code = paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS, models=moved)
    assert code == (3, "")
    assert re.fullmatch(
        r"holdout 0's score \(0, '\S+', 'augmented', '\S+', '\S+'\) .*", error(caplog)
    )


def test_a_replica_fill_comes_from_the_store(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    lines = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if line.get("arm") == "smote-replica" and line["holdout"] == 1:
            line["csv"] = line["csv"].replace("\n", "\r\n", 1)
    store.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    assert paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS) == (3, "")
    assert error(caplog).startswith("holdout 1's smote-replica csv does not hash to its sha256")


def shorting(runner: Runner, *, last: bool) -> Runner:
    """`runner`'s rows but the first cell's, reported short.

    With `last`, every attempt is short, so a gap fill keeps its last attempt's rows. Otherwise
    each call returns its rows whole.
    """

    def short(train: pd.DataFrame, request: Request) -> Baseline:
        full = runner(train, request)
        if not last or not len(full.added):
            return full
        cell, *_ = request.quota
        keep = [stratum != cell for stratum in cell_list(full.added)]
        missing = {"/".join(map(str, cell)): request.quota[cell]}
        kept = Baseline(number(full.added.loc[keep]), full.weights, {"short": missing})
        msg = f"short of {missing}"
        raise ShortfallError(msg, missing, kept.added, kept)

    return short


def arm_lines(caplog, name: str) -> list[str]:
    """Every `arm` line of the arm `name` logged."""
    return [m for m in caplog.messages if m.startswith("arm ") and f" arm={name} " in m]


def reproduces(
    tmp_path: Path, caplog, experiment: Experiment, arms: Mapping[str, Runner], name: str
) -> list[str]:
    """Store `experiment`'s folder, run `panel` with `name` spied; return `name`'s `arm` lines.

    Asserts that `panel` exits 0, never calls `name`'s runner, logs each of its `arm` lines as the
    folder does and logs no `shortfall` line.
    """
    folder, store = folder_and_store(tmp_path, caplog, experiment, arms)
    spy = Spy(arms[name])
    with caplog.at_level(logging.INFO):
        code, _ = paneled(tmp_path, folder, store, experiment=experiment, arms={**arms, name: spy})
    assert code == 0
    assert spy.calls == 0
    logged = sorted(line for key, line in read_folder(folder).arms.items() if key[1] == name)
    assert sorted(arm_lines(caplog, name)) == logged
    assert not [m for m in caplog.messages if m.startswith("shortfall ")]
    return logged


ONLY_CART = dataclasses.replace(
    NARROW, arms=("none", "cart"), candidate="cart", comparators=("none",), criteria=()
)


def test_a_last_attempt_partial_fill_reproduces_its_arm_line(tmp_path, caplog):
    arms = {"none": FAST_ARMS["none"], "cart": shorting(FAST_ARMS["cart"], last=True)}
    logged = reproduces(tmp_path, caplog, ONLY_CART, arms, "cart")
    attempts = ONLY_CART.attempts
    assert all(f" attempt={attempts} " in line and '"short": {' in line for line in logged)


THIN = dataclasses.replace(
    NARROW,
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    criteria=(),
    removal=Removal(("<55", "female", 1), 0.75),
    decontaminate=False,
)
"""A run whose removal leaves `<55 female 1` one training record: SMOTE's shortfall there is
structural."""


def test_a_structural_partial_fill_reproduces_its_arm_line(tmp_path, caplog):
    arms = {"none": FAST_ARMS["none"], "smote": FAST_ARMS["smote"]}
    logged = reproduces(tmp_path, caplog, THIN, arms, "smote")
    assert all(" attempt=1 " in line and '"short": {' in line for line in logged)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("log_sha256", "0" * 64, r"the store's log_sha256 0{64} is not \S+'s \w+"),
        ("folder", "other", r"the store names the folder 'other', not 'run'"),
    ],
)
def test_panel_refuses_a_store_of_another_folder(tmp_path, caplog, field, value, message):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    lines = store.read_text(encoding="utf-8").splitlines(keepends=True)
    header = json.loads(lines[0])
    header[field] = value
    store.write_text(json.dumps(header) + "\n" + "".join(lines[1:]), encoding="utf-8")
    assert paneled(tmp_path, folder, store, experiment=MIXED, arms=MIXED_ARMS) == (3, "")
    assert re.fullmatch(message, error(caplog))


def test_a_stored_baseline_refuses_outside_a_fill(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, MIXED, MIXED_ARMS)
    stored = StoredBaseline("smote", read_rows(store), read_folder(folder))
    with pytest.raises(EvaluationError, match="inside a harness fill"):
        stored(pd.DataFrame(), Request(frame=UCI_FRAME))


def test_admitted_holds_the_panel_and_calibration_measures_only():
    def key(measure: Measure, source: Source = Source.AUGMENTED) -> tuple[int, str, str, str, str]:
        return (0, "none", source.value, measure.value, "all")

    assert admitted(key(Measure.RECALL_RF))
    assert admitted(key(Measure.AUC_PR_NB))
    assert admitted(key(Measure.BRIER))
    assert admitted(key(Measure.CALIBRATION_HGB))
    assert not admitted(key(Measure.RECALL))
    assert not admitted(key(Measure.AUC_PR_HGB))
    assert not admitted(key(Measure.HELD_OUT_RATIO, Source.HELD_OUT))
    assert not admitted(key(Measure.RECALL_RF, Source.REPLICA))
    assert set(MODELS) < set(PANEL)


def test_panel_help_exits_0():
    with pytest.raises(SystemExit) as stopped:
        main(["panel", "--help"])
    assert stopped.value.code == 0


def test_the_context_scores_the_whole_panel_by_default():
    assert Context().panel is PANEL
