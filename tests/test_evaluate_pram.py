"""`bko-direct-pram`: invariant PRAM on `bko-direct`'s unmapped fields, and the check."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import re

import numpy as np
import pandas as pd
import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import digest
from bko.evaluate.direct import DirectArm
from bko.evaluate.experiment import DIRECT_ARM, DIRECT_ARMS, PRAM_ARM
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import (
    DELTA_EMPTY,
    DELTA_THIN,
    THETA,
    PramArm,
    check_distribution,
    passes,
    pram,
    unmapped,
)
from bko.evaluate.run_folder import BKO
from bko.generate.presence import Presence
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


def pram_arm(theta: float = THETA, *, delta: float = 1.0) -> PramArm:
    """The PRAM arm over `varied_arm`'s field rows; a δ of 1 never refuses."""
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=PRAM_ARM, budget=3, sink=None, dataset="uci-test"
    )
    return PramArm(inner, theta, fields=FIELDS, delta=delta)


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


def test_pram_leaves_each_codes_share_invariant_in_expectation():
    train, request, _ = varied_arm()
    coded = ("fbs", "restecg", "exang", "slope", "ca", "thal")
    spread = np.random.default_rng(1)
    train = train.assign(
        **{
            name: spread.choice(sorted(request.frame.codebook[name]), size=len(train))
            for name in coded
        }
    )
    group = train[train["target"] == 1]
    rows = pd.concat([group] * math.ceil(20_000 / len(group))).iloc[:20_000]
    rng = np.random.default_rng(0)
    out = pram(rows, train, fields=FIELDS, theta=THETA, rng=rng, frame=request.frame)
    assert len(out) == 20_000
    changed = 0
    for name in ("fbs", "restecg", "exang", "slope", "ca", "thal"):
        expected = rows[name].value_counts(normalize=True)
        found = out[name].value_counts(normalize=True).reindex(expected.index, fill_value=0.0)
        assert (found - expected).abs().max() <= 0.01, name
        changed += int((out[name] != rows[name]).sum())
    assert changed > 0
    untouched = [column for column in rows.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(out[untouched], rows[untouched])


def test_the_check_refuses_below_the_reference_less_delta_and_accepts_at_equality():
    assert passes(0.90, 0.90, 0.0)
    assert passes(0.89, 0.90, 0.01)
    assert not passes(0.889, 0.90, 0.01)
    assert not passes(math.nan, 0.90, 0.01)
    assert not passes(0.90, math.nan, 0.01)


def test_rows_equal_to_the_originals_pass_the_check_at_delta_zero():
    train, request, direct = varied_arm()
    rows = direct(train, request).added
    found = check_distribution(rows, rows, train, fields=FIELDS, frame=request.frame, delta=0.0)
    assert found.shapes == found.reference
    assert found.passed
    assert 0 < found.pair_trends <= 1


def test_the_deltas_are_not_yet_fixed():
    assert (DELTA_THIN, DELTA_EMPTY) == (0.0, 0.0)


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
    checks = [m for m in caplog.messages if m.startswith("perturb check ")]
    pattern = (
        r"perturb check holdout=(0|1) arm=bko-direct-pram shapes=\d\.\d{6} reference=\d\.\d{6}"
        r" delta=0\.000000 pair_trends=\d\.\d{6} outcome=(accepted|refused)"
    )
    assert len(checks) == 2
    assert all(re.fullmatch(pattern, m) for m in checks), checks


def test_the_stored_arm_rebuilds_the_pram_arms_rows_to_its_arm_sha(tmp_path, caplog):
    _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    argv = ["shared", "g2", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    assert out.getvalue() == (folder / "scores.csv").read_text(encoding="utf-8")
