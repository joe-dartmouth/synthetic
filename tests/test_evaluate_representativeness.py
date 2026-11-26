"""The representativeness reading: each fill against the fused shares of the cells it fills."""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.attack_stats import scopes
from bko.evaluate.data import held_split, load
from bko.evaluate.experiment import REAL_ARM, flagged
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.prepare import prepared
from bko.evaluate.representativeness import REPRESENTATIVENESS_COLUMNS, share_gap
from tests._evaluate_bko_fixtures import NARROW
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_membership import PRIVATE, _context, _header, _rewrite, _stored


def _read(
    tmp_path: Path, *stores: Path, run: str = "e1", runs: dict | None = None
) -> tuple[int, str]:
    argv = ["representativeness", run, "--data-dir", str(tmp_path / "data")]
    for store in stores:
        argv += ["--rows", str(store)]
    out = io.StringIO()
    code = main(argv, context=_context(tmp_path, runs=runs), out=out)
    return code, out.getvalue()


def test_the_gap_is_the_largest_share_difference():
    added = pd.DataFrame(
        {"age": [40, 41, 42], "sex": [0, 0, 0], "cp": [1, 1, 2], "target": [1, 1, 1]}
    )
    # Angina shows in 2 of 3 rows against 1/4: 5/12; stenosis in all 3 against 0.9: 1/10.
    gap, concept = share_gap(
        added, {"HP:9000003": 0.25, "MONDO:9000004": 0.9}, (ANGINA, STENOSIS), name="t"
    )
    assert gap == pytest.approx(5 / 12)
    assert concept == "HP:9000003"
    # Stenosis now sits further: 1 - 0.3 = 0.7 against |2/3 - 0.5| = 1/6.
    gap, concept = share_gap(
        added, {"HP:9000003": 0.5, "MONDO:9000004": 0.3}, (ANGINA, STENOSIS), name="t"
    )
    assert (gap, concept) == (pytest.approx(0.7), "MONDO:9000004")
    empty = share_gap(added.iloc[:0], {"HP:9000003": 0.5}, (ANGINA, STENOSIS), name="t")
    assert math.isnan(empty[0])
    assert empty[1] == ""


def test_the_cell_share_is_read_beside_reals(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    code, text = _read(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text), keep_default_na=False)
    assert tuple(table.columns) == REPRESENTATIVENESS_COLUMNS
    ctx = _context(tmp_path)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    scores = pd.read_csv(folder / "scores.csv")
    for seed in PRIVATE.holdout_seeds:
        ready = prepared(held_split(data, seed), PRIVATE, edges=UCI_FRAME.edges)
        train = ready.ablated.train
        cells = flagged(train, PRIVATE, edges=UCI_FRAME.edges)
        assert cells
        mine = table[table["holdout"] == seed]
        assert len(mine) == len(cells) * len(PRIVATE.arms)
        for cell in cells:
            key = "/".join(map(str, cell))
            rows = mine[mine["cell"] == key]
            by = {str(row["arm"]): row for row in rows.to_dict("records")}
            own = int(scopes(train, cell, edges=UCI_FRAME.edges)["cell"].sum())
            removed = int(scopes(ready.removed, cell, edges=UCI_FRAME.edges)["cell"].sum())
            real = (own + removed) / (len(train) + len(ready.removed))
            assert rows["real_share"].astype(float).tolist() == pytest.approx([real] * 3)
            assert float(by[REAL_ARM]["cell_share"]) == pytest.approx(real)
            assert float(by["none"]["cell_share"]) == pytest.approx(own / len(train))
            assert int(by["none"]["rows"]) == 0
            assert by["none"]["gap"] == ""
            recall = scores[
                (scores["holdout"] == seed)
                & (scores["arm"] == "smote")
                & (scores["source"] == "augmented")
                & (scores["measure"] == "recall")
                & (scores["scope"] == "flagged-subgroups")
            ]["value"]
            assert float(by["smote"]["recall"]) == pytest.approx(float(recall.iloc[0]))
    logged = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("representativeness lines=") for m in logged)


def test_a_store_of_another_run_is_refused(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _header("run", "b1-uci"))
    assert _read(tmp_path, store) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "not a slice of" in errors[-1]


def test_a_run_without_real_is_refused(tmp_path, caplog):
    ctx = _context(tmp_path)
    runs = {**ctx.runs, "plain": NamedRun(NARROW, ctx.settings, ctx.dataset)}
    assert _read(tmp_path, tmp_path / "absent.jsonl", run="plain", runs=runs) == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "lacks 'real'" in errors[-1]
