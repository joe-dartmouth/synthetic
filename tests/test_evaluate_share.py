"""`Experiment.synthetic_share`: a share of each cell's training rows replaced by synthetic rows."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from fractions import Fraction

import pytest

from bko.evaluate.experiment import E1, REAL_ARM, Experiment, Removal
from bko.evaluate.prepare import prepared
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run_folder import ASSUMED
from tests._frame_fixtures import TOY, TOY_PARTS
from tests.test_evaluate_shared import edit_log, refused, run_folder, shared


FULL = ("<45", "male", 1)
"""A toy cell holding 20 training rows at seed 0."""

SHARES = (Fraction(1, 50), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(1))

SHARE_RUN = Experiment(
    holdout_seeds=(0, 1),
    arms=(REAL_ARM, "smote"),
    candidate="smote",
    comparators=(),
    replica_arms=(),
    decontaminate=False,
    flagged_cell=FULL,
    synthetic_share=Fraction(1, 4),
)
"""A toy share run: a quarter of each cell's training rows replaced, `real` beside SMOTE."""


def test_a_share_beside_a_removal_is_refused():
    message = r"a synthetic share cannot stand beside a removal or a training fraction below 1"
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SHARE_RUN, removal=Removal(FULL, 0.5))
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SHARE_RUN, training_fraction=Fraction(1, 2))


@pytest.mark.parametrize("share", [Fraction(0), Fraction(3, 2), 0.25])
def test_a_share_outside_the_unit_interval_is_refused(share):
    with pytest.raises(ValueError, match=r"synthetic share .* must be a Fraction in \(0, 1\]"):
        dataclasses.replace(SHARE_RUN, synthetic_share=share)


def test_a_share_needs_a_flagged_cell():
    with pytest.raises(ValueError, match=r"a synthetic share needs a flagged cell"):
        dataclasses.replace(SHARE_RUN, flagged_cell=None)


def test_real_is_admitted_under_a_share():
    assert SHARE_RUN.arms == (REAL_ARM, "smote")
    with pytest.raises(ValueError, match=r"needs a removal or a synthetic share"):
        dataclasses.replace(SHARE_RUN, synthetic_share=None)


def test_prepared_keeps_the_whole_split_and_names_the_dropped_rows(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.prepare")
    ready = prepared(TOY_PARTS, SHARE_RUN, edges=TOY.edges)
    assert ready.ablated.train.equals(TOY_PARTS.train)
    assert ready.removed.empty
    # Ten cells of 20 training rows drop 5 each and two of 5 drop 1 each.
    assert len(ready.ablated.train) == 210
    assert len(ready.dropped) == 52
    assert set(ready.dropped.index) <= set(TOY_PARTS.train.index)
    assert ready.dropped.equals(TOY_PARTS.train.loc[ready.dropped.index])
    ids = ",".join(str(record) for record in sorted(ready.dropped.index))
    digest = hashlib.sha256(ids.encode("ascii")).hexdigest()
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("share ")]
    assert line == f"share holdout=0 share=1/4 kept=158 dropped=52 dropped_sha256={digest}"


def test_no_share_drops_no_row(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.prepare")
    plain = dataclasses.replace(SHARE_RUN, arms=("none", "smote"), synthetic_share=None)
    ready = prepared(TOY_PARTS, plain, edges=TOY.edges)
    assert ready.dropped.empty
    assert list(ready.dropped.columns) == list(TOY_PARTS.train.columns)
    assert "share holdout=" not in caplog.text


@pytest.mark.parametrize("share", SHARES)
def test_the_test_split_and_its_fingerprint_are_untouched_at_every_share(share):
    experiment = dataclasses.replace(SHARE_RUN, synthetic_share=share)
    ready = prepared(TOY_PARTS, experiment, edges=TOY.edges)
    assert ready.ablated.test.equals(TOY_PARTS.test)
    assert ready.ablated.fingerprint == TOY_PARTS.fingerprint
    assert ready.ablated.train.equals(TOY_PARTS.train)
    assert ready.sized.train.equals(TOY_PARTS.train)


def test_the_experiment_line_names_a_share_only_when_set():
    plain = experiment_line("x", E1, ["lr"], frame="uci")
    assert "synthetic_share" not in plain
    cell = dataclasses.replace(E1, flagged_cell=FULL)
    named = experiment_line("x", cell, ["lr"], frame="uci")
    for share, spelled in [(Fraction(1, 50), "1/50"), (Fraction(1), "1/1")]:
        sharing = dataclasses.replace(cell, synthetic_share=share)
        line = experiment_line("x", sharing, ["lr"], frame="uci")
        assert f" training_fraction=1 synthetic_share={spelled} decontaminate=" in line
        assert line.replace(f"synthetic_share={spelled} ", "") == named


def test_an_older_experiment_line_reads_at_the_assumed_synthetic_share(tmp_path, caplog):
    assert ASSUMED["synthetic_share"] == "none"
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace(
            "training_fraction=1 ", "training_fraction=1 synthetic_share=none ", 1
        ),
    )
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("synthetic_share=none ", "synthetic_share=1/2 ", 1),
    )
    error = refused(tmp_path, caplog, folder)
    assert error.endswith("differ: {'synthetic_share': ('1/2', 'none')}")
