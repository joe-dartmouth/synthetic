"""`bko-direct-pram`: PRAM on `bko-direct`'s unmapped fields, and the reading."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import re
from typing import Final

import numpy as np
import pandas as pd
import pytest

from bko.baselines.request import Stratum
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import perturb
from bko.evaluate import pram as pram_module
from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import digest
from bko.evaluate.direct import DirectArm
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, DIRECT_ARMS, PRAM_ARM
from bko.evaluate.holdout_candidates import CHANGED, KEPT, UNCHANGED, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import (
    DIABETES_UNMAPPED,
    POSITIVE,
    Changed,
    Quality,
    Record,
    label_rows,
    native,
    read_distribution,
    unmapped,
)
from bko.evaluate.pram import PRAM_STREAM, THETA, PramArm, pram
from bko.evaluate.run_folder import BKO
from bko.generate.presence import Presence
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_direct import ADDED_DIGEST, SMALL_G2, g2_context, varied_arm
from tests.test_evaluate_shared import FORMAT


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FIELDS = (
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)
"""UCI's columns that `ANGINA` and `STENOSIS` leave unmapped, in frame order."""

PRAMMED = dataclasses.replace(
    SMALL_G2,
    arms=("none", DIRECT_ARM, PRAM_ARM),
    candidate=PRAM_ARM,
    comparators=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Two holdouts, one baseline, `bko-direct` and `bko-direct-pram`, the PRAM arm the candidate."""

PRAM_DIGEST: Final = "1982f7231e5e798b180db7509983e14ca21980ed90716ad6421c5c3e1ea3f3bc"
"""`pram` of `varied_arm`'s draw over `FIELDS` at theta 0.5 from `default_rng(7)`: pins each
field's keep draw, then its pick draw, field by field in column order."""


def pram_arm(theta: float = THETA, *, fields: tuple[str, ...] = FIELDS) -> PramArm:
    """The PRAM arm over `varied_arm`'s field rows."""
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=PRAM_ARM, budget=3, sink=None, dataset="uci-test"
    )
    return PramArm(inner, theta, fields=fields)


def test_the_diabetes_unmapped_fields_are_the_pinned_19():
    assert unmapped(DIABETES_FIELD_ROWS, frame=DIABETES_FRAME) == DIABETES_UNMAPPED
    assert len(DIABETES_UNMAPPED) == 19
    with pytest.raises(EvaluationError, match="not its pinned"):
        unmapped(DIABETES_FIELD_ROWS[1:], frame=DIABETES_FRAME)


@pytest.mark.parametrize("field", ["age", "sex", "target", "cp"])
def test_the_arm_refuses_to_perturb_the_cell_the_label_or_a_mapped_field(field):
    train, request, _ = varied_arm()
    with pytest.raises(EvaluationError, match="may not perturb"):
        pram_arm(fields=(*FIELDS, field))(train, request)


def test_the_unmapped_fields_are_every_column_no_field_row_reads_but_age_sex_and_label():
    _, request, _ = varied_arm()
    assert unmapped((ANGINA, STENOSIS), frame=request.frame) == FIELDS


def test_the_pram_arm_is_a_direct_arm_drawing_at_shares_and_a_bko_arm():
    assert DIRECT_ARMS[PRAM_ARM] is Presence.SHARES
    assert PRAM_ARM in BKO


def test_pram_changes_only_the_unmapped_fields():
    train, request, direct = varied_arm()
    ours = pram_arm()(train, request)
    theirs = direct(train, request).added
    assert list(ours.added.index) == list(theirs.index)
    kept = [column for column in theirs.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(ours.added[kept], theirs[kept])
    assert (ours.added[list(FIELDS)] != theirs[list(FIELDS)]).to_numpy().any()
    assert ours.params == {"generator": "direct", "seed": 0, "perturb": "pram", "theta": THETA}


def test_the_inner_draw_is_bko_directs_rows():
    train, request, _ = varied_arm()
    assert digest(pram_arm(1.0)(train, request).added) == ADDED_DIGEST


def test_pram_is_seeded_by_the_request_and_repeatable():
    train, request, _ = varied_arm()
    first = pram_arm()(train, request).added
    again = pram_arm()(train, request).added
    other = pram_arm()(train, dataclasses.replace(request, seed=1)).added
    pd.testing.assert_frame_equal(first, again)
    assert digest(first) != digest(other)


@pytest.mark.parametrize("seed", [0, 3])
def test_pram_draws_from_its_own_stream_at_the_request_seed(seed):
    train, request, direct = varied_arm(seed)
    drawn = direct(train, request).added.reset_index(drop=True)
    rng = np.random.default_rng([seed, PRAM_STREAM])
    expected = pram(drawn, train, fields=FIELDS, theta=THETA, rng=rng, frame=request.frame)
    found = pram_arm()(train, request).added.reset_index(drop=True)
    pd.testing.assert_frame_equal(found, expected, check_dtype=False)
    shared = pram(
        drawn,
        train,
        fields=FIELDS,
        theta=THETA,
        rng=np.random.default_rng(seed),
        frame=request.frame,
    )
    assert digest(shared) != digest(expected)


def test_pram_draws_keep_then_pick_field_by_field():
    train, request, direct = varied_arm()
    drawn = direct(train, request).added.reset_index(drop=True)
    out = pram(
        drawn, train, fields=FIELDS, theta=0.5, rng=np.random.default_rng(7), frame=request.frame
    )
    assert digest(out) == PRAM_DIGEST


def test_pram_keeps_a_value_at_theta_plus_its_redraw_matching():
    shares = {3: 0.5, 6: 0.2, 7: 0.3}
    spread = np.random.default_rng(2)
    train = pd.DataFrame(
        {"thal": spread.choice(list(shares), size=10_000, p=list(shares.values())), "target": 1}
    )
    rows = pd.DataFrame(
        {"thal": spread.choice(list(shares), size=20_000, p=list(shares.values())), "target": 1}
    )
    out = pram(
        rows, train, fields=("thal",), theta=THETA, rng=np.random.default_rng(0), frame=UCI_FRAME
    )
    found = train["thal"].value_counts(normalize=True)
    expected = THETA + (1 - THETA) * float((found**2).sum())
    assert float((out["thal"] == rows["thal"]).mean()) == pytest.approx(expected, abs=0.01)


def test_pram_redraws_each_field_toward_its_label_groups_training_marginal():
    """Rows drawn from each label group's training marginal keep it after PRAM, coded and
    measured fields alike, though the two groups' marginals differ: within 4 standard errors."""
    spread = np.random.default_rng(1)
    size = 4_000
    train = pd.DataFrame(
        {
            "thal": np.concatenate(
                [
                    spread.choice([3, 6, 7], size=size, p=[0.7, 0.2, 0.1]),
                    spread.choice([3, 6, 7], size=size, p=[0.2, 0.3, 0.5]),
                ]
            ),
            "chol": np.concatenate(
                [spread.normal(200, 20, size).round(), spread.normal(260, 40, size).round()]
            ),
            "target": [0] * size + [1] * size,
        }
    )
    rows = train.sample(n=20_000, replace=True, random_state=3).reset_index(drop=True)
    out = pram(
        rows,
        train,
        fields=("thal", "chol"),
        theta=THETA,
        rng=np.random.default_rng(0),
        frame=UCI_FRAME,
    )
    for label in (0, 1):
        before, after = rows[rows["target"] == label], out[out["target"] == label]
        group = train[train["target"] == label]
        n = len(before)
        for code, share in group["thal"].value_counts(normalize=True).items():
            sigma = math.sqrt(share * (1 - share) / n)
            assert abs(float((after["thal"] == code).mean()) - share) <= 4 * sigma, (label, code)
        sigma = float(group["chol"].std()) / math.sqrt(n)
        assert abs(float(after["chol"].mean()) - float(group["chol"].mean())) <= 4 * sigma, label
        assert (after["thal"] != before["thal"]).any()
        assert (after["chol"] != before["chol"]).any()
    pd.testing.assert_series_equal(out["target"], rows["target"])


def test_a_nonfinite_shape_is_read_and_logged_nan(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    train, request, direct = varied_arm()
    rows = direct(train, request).added
    cell: Stratum = ("x", "male", 1)
    records = [native(row, request.frame) for row in rows.to_dict("records")]
    drawn: dict[Stratum, list[Record]] = {cell: records}
    changed: dict[Stratum, list[Changed]] = {cell: [Changed(dict(row)) for row in records]}
    nan = Quality(math.nan, math.nan, dict.fromkeys(FIELDS, math.nan))
    monkeypatch.setattr(perturb, "quality", lambda *_, **__: nan)
    arm = pram_arm()
    assert arm.read(train, drawn, changed, request.frame) is None
    (line,) = [m for m in caplog.messages if m.startswith("perturb shapes ")]
    assert " shapes=nan unperturbed=nan difference=nan pair_trends=nan " in line
    assert arm.counts["read"] == 1


def test_rows_equal_to_the_originals_read_no_difference():
    train, request, direct = varied_arm()
    rows = direct(train, request).added
    reference = label_rows(train, request.frame, POSITIVE)
    found = read_distribution(rows, rows, reference, fields=FIELDS, frame=request.frame)
    assert found.perturbed == found.unperturbed
    assert found.difference == pytest.approx(0.0)
    assert 0 < found.perturbed.pair_trends <= 1
    assert tuple(found.perturbed.columns) == FIELDS
    assert (found.incoherent, found.unperturbed_incoherent) == (None, None)


def test_the_reading_reads_only_its_fields_against_the_reference():
    train, request, direct = varied_arm()
    rows = direct(train, request).added
    frame = request.frame
    reference = label_rows(train, frame, POSITIVE)
    matched = reference.reset_index(drop=True)
    found = read_distribution(matched, rows, reference, fields=("chol",), frame=frame)
    assert found.perturbed.shapes == pytest.approx(1.0)
    assert found.unperturbed.shapes < 1.0
    assert found.difference > 0
    everyone = read_distribution(train, rows, reference, fields=("chol",), frame=frame)
    assert everyone.perturbed.shapes < 1.0
    skewed = matched.assign(trestbps=matched["trestbps"].min())
    alone = read_distribution(skewed, rows, reference, fields=("chol",), frame=frame)
    assert alone.perturbed.shapes == pytest.approx(1.0)
    both = read_distribution(skewed, rows, reference, fields=("trestbps", "chol"), frame=frame)
    assert both.perturbed.shapes < 1.0
    assert both.perturbed.columns["trestbps"] < both.perturbed.columns["chol"]


def test_a_fill_that_drew_no_row_is_not_read(caplog):
    caplog.set_level(logging.INFO)
    train, request, _ = varied_arm()
    arm = pram_arm()
    assert arm.read(train, {}, {}, request.frame) is None
    assert arm.counts["empty"] == 1
    shapes = [m for m in caplog.messages if m.startswith("perturb shapes ")]
    assert shapes == ["perturb shapes holdout=-1 arm=bko-direct-pram empty=true"]
    assert not [m for m in caplog.messages if m.startswith("perturb columns ")]


def test_a_summary_with_no_read_holdout_writes_no_mean(caplog):
    caplog.set_level(logging.INFO)
    train, request, _ = varied_arm()
    arm = pram_arm()
    arm.read(train, {}, {}, request.frame)
    arm.summary()
    (summary,) = [m for m in caplog.messages if m.startswith("perturb summary ")]
    assert summary.endswith(
        " read=0 empty=1 rows=0 rows_changed=0 values_changed=0"
        " mean_shapes=none mean_unperturbed=none"
    )


def test_a_field_sdmetrics_does_not_detail_reads_nan_and_warns(caplog, monkeypatch):
    class Report:
        def get_properties(self) -> pd.DataFrame:
            return pd.DataFrame(
                {"Property": ["Column Shapes", "Column Pair Trends"], "Score": [0.5, 0.25]}
            )

        def get_details(self, _: str) -> pd.DataFrame:
            return pd.DataFrame({"Column": ["chol"], "Score": [0.75]})

    monkeypatch.setattr(perturb, "quality_report", lambda *_, **__: Report())
    caplog.set_level(logging.INFO)
    train, _, _ = varied_arm()
    found = perturb.quality(train, train, fields=("chol", "trestbps"), frame=UCI_FRAME)
    assert found.columns["chol"] == pytest.approx(0.75)
    assert math.isnan(found.columns["trestbps"])
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warned] == ["perturb unscored holdout=-1 field=trestbps"]


def test_a_pram_reading_below_the_unperturbed_keeps_the_arms_rows(caplog, monkeypatch):
    """A redraw that sets every field to its least value reads below the drawn rows, and the arm
    keeps it."""
    caplog.set_level(logging.INFO)
    train, request, direct = varied_arm()
    drawn = direct(train, request).added.reset_index(drop=True)

    def least(
        rows: pd.DataFrame, *_: object, fields: tuple[str, ...], **__: object
    ) -> pd.DataFrame:
        return rows.assign(**{name: rows[name].min() for name in fields})

    monkeypatch.setattr(pram_module, "pram", least)
    arm = pram_arm()
    found = arm(train, request).added.reset_index(drop=True)
    pd.testing.assert_frame_equal(found, least(drawn, fields=FIELDS), check_dtype=False)
    assert digest(found) != ADDED_DIGEST
    (line,) = [m for m in caplog.messages if m.startswith("perturb shapes ")]
    assert float(line.split(" difference=")[1].split(" ")[0]) < 0, line
    assert arm.counts["rows_changed"] > 0


def _prammed(tmp_path, caplog) -> tuple[pd.DataFrame, Context]:
    caplog.set_level(logging.INFO)
    ctx, send = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(PRAMMED)})
    folder = tmp_path / "run"
    folder.mkdir()
    out = io.StringIO()
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    caplog.clear()
    assert main([*argv, "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    assert send.calls == []
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    return pd.read_csv(io.StringIO(out.getvalue())), ctx


def test_a_score_run_with_the_pram_arm_writes_its_candidates(tmp_path, caplog):
    scores, _ = _prammed(tmp_path, caplog)
    assert PRAM_ARM in set(scores[scores["source"] == "candidates"]["arm"])
    found = read((tmp_path / "run" / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == PRAM_ARM]
    assert len(mine) == 2 * 28
    assert {line.call for line in mine} == {None}
    assert {line.outcome for line in mine} == {KEPT}
    shapes = [m for m in caplog.messages if m.startswith("perturb shapes ")]
    number = r"-?\d\.\d{6}"
    pattern = (
        rf"perturb shapes holdout=(0|1) arm=bko-direct-pram shapes={number}"
        rf" unperturbed={number} difference={number} pair_trends={number}"
        rf" unperturbed_pair_trends={number} incoherent=none unperturbed_incoherent=none"
        r" rows_changed=\d+ values_changed=\d+"
    )
    assert len(shapes) == 2
    assert all(re.fullmatch(pattern, m) for m in shapes), shapes
    columns = [m for m in caplog.messages if m.startswith("perturb columns ")]
    assert len(columns) == 4
    kinds = [m.split(" kind=")[1].split(" ")[0] for m in columns]
    assert kinds == ["perturbed", "unperturbed"] * 2
    prammed = [m for m in caplog.messages if m.startswith("perturb pram ")]
    assert len(prammed) == 2
    assert all(
        re.fullmatch(
            r"perturb pram holdout=(0|1) arm=bko-direct-pram rows=28",
            m,
        )
        for m in prammed
    ), prammed
    (configured,) = [m for m in caplog.messages if m.startswith("perturb arm=")]
    assert configured == (f"perturb arm=bko-direct-pram fields={','.join(FIELDS)} theta=0.5")
    (summary,) = [m for m in caplog.messages if m.startswith("perturb summary ")]
    assert re.fullmatch(
        r"perturb summary arm=bko-direct-pram holdouts=2 read=2 empty=0 rows=56"
        r" rows_changed=\d+ values_changed=\d+ mean_shapes=\d\.\d{6} mean_unperturbed=\d\.\d{6}",
        summary,
    )


def test_the_stored_arm_rebuilds_the_pram_arms_rows_to_its_arm_sha(tmp_path, caplog):
    _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    argv = ["shared", "g2", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    assert out.getvalue() == (folder / "scores.csv").read_text(encoding="utf-8")


def test_every_pram_line_carries_changed_or_unchanged(tmp_path, caplog):
    _prammed(tmp_path, caplog)
    found = read((tmp_path / "run" / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == PRAM_ARM]
    assert {line.perturb for line in mine} == {"changed", "unchanged"}
    for holdout in (0, 1):
        changed = sum(line.perturb == "changed" for line in mine if line.holdout == holdout)
        (shapes,) = [
            m
            for m in caplog.messages
            if m.startswith(f"perturb shapes holdout={holdout} arm={PRAM_ARM} ")
        ]
        assert int(shapes.split(" rows_changed=")[1].split(" ")[0]) == changed


def test_shapes_counts_rows_from_the_lines_flags(tmp_path, caplog, monkeypatch):
    def unflagged(record: Record, *, result: str) -> Changed:
        return Changed(record, result=UNCHANGED)

    monkeypatch.setattr(pram_module, "Changed", unflagged)
    _prammed(tmp_path, caplog)
    prefix = f"perturb shapes holdout=0 arm={PRAM_ARM} "
    (line,) = [m for m in caplog.messages if m.startswith(prefix)]
    assert " rows_changed=0 " in line, line
    assert int(line.split(" values_changed=")[1]) > 0, line


def test_rows_changed_is_above_zero_exactly_when_values_changed_is(tmp_path, caplog):
    _prammed(tmp_path, caplog)
    lines = [m for m in caplog.messages if m.startswith("perturb shapes holdout=")]
    assert lines
    for line in lines:
        rows = int(line.split(" rows_changed=")[1].split(" ")[0])
        values = int(line.split(" values_changed=")[1])
        assert (rows > 0) == (values > 0), line


@pytest.mark.parametrize("field", FIELDS)
def test_a_row_whose_only_change_is_one_field_reads_changed(field, monkeypatch):
    train, request, _ = varied_arm()

    def one(rows: pd.DataFrame, *_: object, **__: object) -> pd.DataFrame:
        out = rows.copy()
        values = out[field].to_numpy(copy=True)
        values[0] += 1
        out[field] = values
        return out

    monkeypatch.setattr(pram_module, "pram", one)
    monkeypatch.setattr(PramArm, "read", lambda *_: None)
    records = [native(row, request.frame) for row in train.head(3).to_dict("records")]
    drawn: dict[Stratum, list[Record]] = {("<55", "female", 0): records}
    changed = pram_arm()._change(train, request, drawn)
    assert [row.result for row in changed["<55", "female", 0]] == [
        CHANGED,
        UNCHANGED,
        UNCHANGED,
    ]
