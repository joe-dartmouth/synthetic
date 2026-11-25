"""A synthetic-share run's folder read end to end: `score`, then `rows`, `panel`, `shared`,
`merge`, `trace` and `privacy`, each rebuilding the run's request from the whole training split."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Final

import pytest

import bko.evaluate.reference as reference_module
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import DIRECT_ARM, REAL_ARM
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.scores import Measure, Scope, Source, parse
from tests._evaluate_bko_fixtures import NARROW
from tests._evaluate_fixtures import FAST_ARMS
from tests.test_evaluate_direct import g2_context
from tests.test_evaluate_panel import FAST_PANEL
from tests.test_evaluate_trace import _scored


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CELL: Final = ("<55", "female", 1)
"""`VARIED`'s young women with the condition: 4 training rows on every holdout."""

SHARE: Final = dataclasses.replace(
    NARROW,
    holdout_seeds=(0, 1, 2, 3),
    arms=(REAL_ARM, "smote", DIRECT_ARM),
    candidate=DIRECT_ARM,
    comparators=("smote",),
    criteria=(),
    decontaminate=False,
    flagged_cell=CELL,
    synthetic_share=Fraction(1, 2),
)
"""Half of each cell's training rows replaced, `bko-direct` and SMOTE beside `real`."""

RUNS: Final = {
    "whole": SHARE,
    "whole-a": dataclasses.replace(SHARE, holdout_seeds=(0, 1)),
    "whole-b": dataclasses.replace(SHARE, holdout_seeds=(2, 3)),
}
"""The share run and its two slices."""


def context() -> Context:
    """`VARIED`'s world with the share runs registered; any call fails."""
    ctx, _ = g2_context()
    runs = {name: NamedRun(e, ctx.settings, ctx.dataset) for name, e in RUNS.items()}
    return dataclasses.replace(
        ctx, experiment=SHARE, arms={"smote": FAST_ARMS["smote"]}, runs={**ctx.runs, **runs}
    )


def _data(root: Path) -> list[str]:
    return ["--data-dir", str(root / "data")]


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """The share run's folder, each slice's folder, and the run's `rows` store."""
    base = tmp_path_factory.mktemp("share")
    for name in RUNS:
        folder = base / name
        argv = ["score", name, "--candidates", str(folder / "holdout-candidates.jsonl")]
        _scored(folder, context(), [*argv, *_data(base)])
    (base / "store").mkdir()
    argv = [
        "rows",
        "whole",
        "--folder",
        str(base / "whole"),
        "--rows",
        str(base / "store" / "rows.jsonl"),
    ]
    assert main([*argv, *_data(base)], context=context(), out=io.StringIO()) == 0
    return base


def _main(argv: list[str], ctx: Context | None = None) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, context=context() if ctx is None else ctx, out=out)
    return code, out.getvalue()


def test_score_logs_each_holdouts_share_and_asks_each_cells_dropped_rows(root):
    log = (root / "whole" / "scores.log").read_text(encoding="utf-8")
    assert " training_fraction=1 synthetic_share=1/2 decontaminate=false " in log
    for seed in SHARE.holdout_seeds:
        assert f"share holdout={seed} share=1/2 kept=66 dropped=66 dropped_sha256=" in log
        quota = {
            line.split("stratum=")[1].split(" ")[0]: int(line.rsplit("rows=", 1)[1])
            for line in log.splitlines()
            if f"quota holdout={seed} " in line
        }
        assert quota == {
            "<55/female/0": 4,
            "<55/female/1": 2,
            **{f"{band}/{sex}/{label}": 10 for band, sex, label in _full_cells()},
        }


def _full_cells() -> list[tuple[str, str, int]]:
    """`VARIED`'s cells of 20 training rows."""
    return [
        (band, sex, label)
        for band in ("<55", ">=55")
        for sex in ("female", "male")
        for label in (0, 1)
        if (band, sex) != ("<55", "female")
    ]


def test_rows_stores_every_fill_of_a_share_run(root):
    lines = (root / "store" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    arms = {(line["holdout"], line["arm"]) for line in map(json.loads, lines[1:])}
    assert arms == {(seed, arm) for seed in SHARE.holdout_seeds for arm in SHARE.arms}


def test_panel_scores_a_share_runs_stored_fills(root):
    argv = ["panel", "whole", "--folder", str(root / "whole")]
    argv += ["--rows", str(root / "store" / "rows.jsonl"), *_data(root)]
    code, text = _main(argv, dataclasses.replace(context(), panel=FAST_PANEL))
    assert code == 0
    measures = {s.measure for s in parse(text)}
    assert {Measure("auc-roc-rf"), Measure("recall-nb")} <= measures


def test_shared_rewrites_a_share_runs_scores_bytes(root):
    code, text = _main(["shared", "whole", "--folder", str(root / "whole"), *_data(root)])
    assert code == 0
    assert text == (root / "whole" / "scores.csv").read_text(encoding="utf-8")


def test_merging_a_share_runs_slices_rewrites_its_scores_bytes(root):
    paths = [str(root / name / "scores.csv") for name in ("whole-a", "whole-b")]
    code, text = _main(["merge", "whole", *paths])
    assert code == 0
    assert text == (root / "whole" / "scores.csv").read_text(encoding="utf-8")


def test_trace_redraws_a_share_runs_direct_fill(root, tmp_path):
    argv = ["trace", "whole", "--folder", str(root / "whole"), "--holdout", "1"]
    code, text = _main([*argv, "--arm", DIRECT_ARM, *_data(root)])
    assert code == 0
    # Half of each cell's training rows: 10 of each full cell, 4 and 2 of the young women's.
    assert len(text.splitlines()) == 6 * 10 + 4 + 2


def test_privacy_reads_a_share_run_at_its_flagged_cell(root, monkeypatch, caplog):
    monkeypatch.setattr(
        reference_module, "REFERENCE", {**reference_module.REFERENCE, "whole": DIRECT_ARM}
    )
    caplog.set_level(logging.INFO)
    argv = ["privacy", "whole", "--rows", str(root / "store" / "rows.jsonl"), *_data(root)]
    code, text = _main(argv, dataclasses.replace(context(), runs_dir=root))
    assert code == 0
    scores = parse(text)
    assert {s.holdout for s in scores} == set(SHARE.holdout_seeds)
    assert "against=bko-direct" in caplog.text
    cell = [
        s
        for s in scores
        if s.arm == "smote" and s.source is Source.ADDED and s.measure.value == "mi-auc-cell"
    ]
    assert len(cell) == len(SHARE.holdout_seeds)
    assert all(s.scope is Scope.ALL for s in cell)


def test_contradictions_counts_a_share_runs_fills_against_its_request(root):
    argv = ["contradictions", "whole", "--rows", str(root / "store" / "rows.jsonl"), *_data(root)]
    code, text = _main(argv, dataclasses.replace(context(), runs_dir=root))
    assert code == 0
    rows = [line.split(",") for line in text.splitlines()[1:]]
    empty = {(row[0], row[1]): row[4] for row in rows if row[3] == "empty-cell"}
    # Each arm that answers the request fills every cell it asks; `real` answers none.
    for seed in SHARE.holdout_seeds:
        assert empty[str(seed), "smote"] == "0"
        assert empty[str(seed), DIRECT_ARM] == "0"
        assert empty[str(seed), REAL_ARM] == ""
