"""`Experiment.flagged_cell`: one cell flagged whatever its count and restored to `fill_to`."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, Experiment, Removal, flagged, gap_request
from bko.evaluate.prepare import require_cells
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run import ARMS, run
from bko.evaluate.run_folder import ASSUMED
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN, toy_records
from tests.test_evaluate_shared import edit_log, refused, run_folder, shared


FULL = ("<45", "male", 1)
"""A toy cell holding 20 training rows at seed 0: at the threshold, so not flagged by count."""

TOY_RUN = Experiment(
    holdout_seeds=(0, 1),
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    replica_arms=(),
)
"""A toy experiment that flags by count alone."""


def test_a_flagged_cell_is_flagged_above_the_threshold():
    assert flagged(TOY_TRAIN, TOY_RUN, edges=TOY.edges) == SHORT
    named = dataclasses.replace(TOY_RUN, flagged_cell=FULL)
    assert flagged(TOY_TRAIN, named, edges=TOY.edges) == (FULL,)


def test_the_flagged_cell_is_restored_to_fill_to():
    by_count = gap_request(TOY_TRAIN, dataclasses.replace(TOY_RUN, fill_to=26), seed=0, frame=TOY)
    assert by_count.quota == {SHORT[0]: 21, SHORT[1]: 21}
    named = dataclasses.replace(TOY_RUN, fill_to=26, flagged_cell=FULL)
    assert gap_request(TOY_TRAIN, named, seed=0, frame=TOY).quota == {FULL: 6}


def test_a_flagged_cell_other_than_the_removals_is_refused():
    removal = Removal(FULL, 0.5)
    kept = dataclasses.replace(TOY_RUN, removal=removal, flagged_cell=FULL)
    assert kept.flagged_cell == FULL
    with pytest.raises(ValueError, match=r"flagged cell .*'female'.* is not the removal's cell"):
        dataclasses.replace(TOY_RUN, removal=removal, flagged_cell=("<45", "female", 1))


def test_a_cell_already_at_fill_to_is_refused():
    at = dataclasses.replace(TOY_RUN, fill_to=20, flagged_cell=FULL)
    with pytest.raises(EvaluationError, match=r"holds 20 training records, so fill_to 20 adds no"):
        gap_request(TOY_TRAIN, at, seed=0, frame=TOY)
    absent = dataclasses.replace(TOY_RUN, fill_to=26, flagged_cell=("90+", "male", 1))
    with pytest.raises(EvaluationError, match=r"is not a cell of the frame toy"):
        gap_request(TOY_TRAIN, absent, seed=0, frame=TOY)


def test_the_experiment_line_names_a_flagged_cell_only_when_set():
    plain = experiment_line("x", E1, ["lr"], frame="uci")
    assert "flagged_cell" not in plain
    named = experiment_line("x", dataclasses.replace(E1, flagged_cell=FULL), ["lr"], frame="uci")
    assert " removal=none flagged_cell=<45/male/1 training_fraction=1 " in named
    assert named.replace("flagged_cell=<45/male/1 ", "") == plain


def test_an_older_experiment_line_reads_at_the_assumed_flagged_cell(tmp_path, caplog):
    assert ASSUMED["flagged_cell"] == "none"
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("removal=none ", "removal=none flagged_cell=none ", 1),
    )
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("flagged_cell=none ", "flagged_cell=<45/male/1 ", 1),
    )
    error = refused(tmp_path, caplog, folder)
    assert error.endswith("differ: {'flagged_cell': ('<45/male/1', 'none')}")


def test_a_flagged_cell_the_frame_lacks_is_refused_before_any_holdout():
    absent = dataclasses.replace(TOY_RUN, fill_to=26, flagged_cell=("90+", "male", 1))
    message = r"the flagged cell \('90\+', 'male', 1\) is not a cell of the toy frame"
    with pytest.raises(EvaluationError, match=message):
        require_cells(absent, TOY)
    arms = {name: ARMS[name] for name in absent.arms}
    with pytest.raises(EvaluationError, match=message):
        run(toy_records(), absent, arms, frame=TOY)
    require_cells(dataclasses.replace(absent, flagged_cell=FULL), TOY)
