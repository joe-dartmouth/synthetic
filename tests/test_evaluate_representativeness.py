"""The representativeness reading: each fill against the fused shares of the cells it fills."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate import representativeness as representativeness_module
from bko.evaluate.__main__ import main
from bko.evaluate.attack_stats import scopes
from bko.evaluate.data import held_split, load
from bko.evaluate.experiment import REAL_ARM, flagged
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.prepare import prepared
from bko.evaluate.representativeness import REPRESENTATIVENESS_COLUMNS, share_gap
from bko.gaps.quota import fused_shares
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


def test_a_fill_below_the_fused_share_reads_the_absolute_gap():
    added = pd.DataFrame(
        {"age": [40, 41, 42], "sex": [0, 0, 0], "cp": [1, 1, 2], "target": [1, 1, 1]}
    )
    # Angina shows in 2 of 3 rows against 0.9: |2/3 - 0.9| = 7/30; stenosis in all 3 against 1.
    gap, concept = share_gap(
        added, {"HP:9000003": 0.9, "MONDO:9000004": 1.0}, (ANGINA, STENOSIS), name="t"
    )
    assert (gap, concept) == (pytest.approx(7 / 30), "HP:9000003")


def test_no_fused_concept_reads_nan():
    added = pd.DataFrame({"age": [40], "sex": [0], "cp": [1], "target": [1]})
    gap, concept = share_gap(added, {}, (ANGINA, STENOSIS), name="t")
    assert math.isnan(gap)
    assert concept == ""


def test_a_tied_gap_goes_to_the_first_concept_in_sorted_order():
    added = pd.DataFrame({"age": [40, 41], "sex": [0, 0], "cp": [1, 2], "target": [1, 1]})
    # Angina shows in 1 of 2 rows against 0.25, stenosis in both against 0.75: each 1/4 off.
    fused = {"MONDO:9000004": 0.75, "HP:9000003": 0.25}
    assert share_gap(added, fused, (ANGINA, STENOSIS), name="t") == (0.25, "HP:9000003")


def _fills(store: Path, seed: int, arm: str) -> pd.DataFrame:
    """The added rows `store` holds for `arm`'s gap fill at holdout `seed`."""
    lines = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    (line,) = [
        line
        for line in lines[1:]
        if (line["kind"], line["holdout"], line["arm"]) == ("rows", seed, arm)
    ]
    return pd.read_csv(io.StringIO(line["csv"]))


def test_a_row_adding_arms_gap_reads_its_rows_in_the_cell_alone(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    code, text = _read(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text), keep_default_na=False)
    ctx = _context(tmp_path)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    for seed in PRIVATE.holdout_seeds:
        train = prepared(held_split(data, seed), PRIVATE, edges=UCI_FRAME.edges).ablated.train
        cells = flagged(train, PRIVATE, edges=UCI_FRAME.edges)
        fused = fused_shares(train, cells, name=data.name, rows=data.field_rows, frame=UCI_FRAME)
        added = _fills(store, seed, "smote")
        inside_total = 0
        for cell in cells:
            band, sex, label = cell
            young = added["age"] < UCI_FRAME.edges[0]
            inside = added.loc[
                (young if band == f"<{UCI_FRAME.edges[0]}" else ~young)
                & (added["sex"] == {"female": 0, "male": 1}[str(sex)])
                & (added["target"] == label)
            ]
            inside_total += len(inside)
            (row,) = table[
                (table["holdout"] == seed)
                & (table["arm"] == "smote")
                & (table["cell"] == "/".join(map(str, cell)))
            ].to_dict("records")
            assert int(row["rows"]) == len(inside)
            gap, concept = share_gap(inside, fused[cell], data.field_rows, name=data.name)
            assert (float(row["gap"]), row["concept"]) == (pytest.approx(gap), concept)
            assert concept
        assert 0 < inside_total <= len(added)


def test_an_arm_with_no_recall_score_reads_nan(tmp_path, caplog, monkeypatch):
    _, store = _stored(tmp_path, caplog)
    real = representativeness_module.folder_scores

    def without_smote(folder):
        return [score for score in real(folder) if score.arm != "smote"]

    monkeypatch.setattr(representativeness_module, "folder_scores", without_smote)
    code, text = _read(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text))
    assert table.loc[table["arm"] == "smote", "recall"].isna().all()
    assert table.loc[table["arm"] == "none", "recall"].notna().all()


def test_a_swapped_scores_file_is_refused(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    path = folder / "scores.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",1.0\n", ",0.0\n"), encoding="utf-8")
    assert _read(tmp_path, store) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "is not its scores line's" in errors[-1]


def test_a_folder_without_scores_is_refused(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    (folder / "scores.csv").unlink()
    assert _read(tmp_path, store) == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "holds no scores.csv" in errors[-1]


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
            for arm in PRIVATE.arms:
                (recall,) = scores[
                    (scores["holdout"] == seed)
                    & (scores["arm"] == arm)
                    & (scores["source"] == "augmented")
                    & (scores["measure"] == "recall")
                    & (scores["scope"] == "flagged-subgroups")
                ]["value"]
                assert float(by[arm]["recall"]) == pytest.approx(float(recall))
    logged = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("representativeness lines=") for m in logged)
    sha = hashlib.sha256((folder / "scores.csv").read_bytes()).hexdigest()
    assert f"representativeness scores folder={folder} sha256={sha}" in logged


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
