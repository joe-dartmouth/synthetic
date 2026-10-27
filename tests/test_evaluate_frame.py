"""The baselines and the harness follow the frame they are given: a toy frame runs end to end."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from bko.baselines.errors import OutputContractError
from bko.baselines.methods import augment
from bko.baselines.records import Baseline, Drop, check, drops, no_rows, training_set
from bko.baselines.request import Method, Request, plan
from bko.baselines.synthesizers import METADATA, metadata
from bko.datasets.cells import cell_counts
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import fidelity as fidelity_module
from bko.evaluate.classifier import build, probabilities
from bko.evaluate.copies import copies, decontaminated
from bko.evaluate.experiment import Experiment, flagged, gap_request, replica_request
from bko.evaluate.fidelity import fidelity
from bko.evaluate.privacy import distances
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.run import ARMS, held_out_ratio, run
from bko.evaluate.scores import Measure, Scope, Source
from bko.evaluate.utility import in_scope, utility
from bko.generate.messages import SETTINGS, Call
from tests._evaluate_fixtures import FAST_METHODS, GRADED, SMALL
from tests._frame_fixtures import SHORT, TOY, TOY_PARTS, TOY_TRAIN, toy_records


TOY_EXPERIMENT = Experiment(
    holdout_seeds=(0, 1),
    arms=("none", "class-weight", "smote"),
    candidate="smote",
    comparators=("none", "class-weight"),
    replica_arms=("smote",),
)


def toy_added(rows: list[dict[str, int | float]]) -> pd.DataFrame:
    added = pd.DataFrame(rows).astype(dict(TOY.dtypes))
    added.index = pd.Index(range(-1, -len(added) - 1, -1), name="record_id", dtype="int64")
    return added


def test_the_split_and_the_counts_read_the_toy_edges():
    counts = cell_counts(TOY_TRAIN, edges=TOY.edges)
    assert counts["age_band"].unique().tolist() == ["<45", "45-69", ">=70"]
    short = counts.loc[counts["underrepresented"]]
    assert list(zip(short["age_band"], short["sex"], short["target"], strict=True)) == list(SHORT)


def test_a_request_keys_its_quota_by_the_toy_cells():
    request = Request(quota={SHORT[0]: 2}, frame=TOY)
    assert request.frame is TOY
    with pytest.raises(ValueError, match="not a positive row count"):
        Request(quota={("<55", "female", 1): 2}, frame=TOY)
    assert plan(TOY_TRAIN, 3, frame=TOY).frame is TOY
    assert Request().frame is UCI_FRAME


def test_drops_read_the_toy_ranges_and_cells():
    request = Request(quota={SHORT[0]: 1}, frame=TOY)
    rows = toy_added(
        [
            {"age": 50, "sex": 0, "level": 1.25, "target": 1},
            {"age": 50, "sex": 0, "level": 9.0, "target": 1},
            {"age": 36, "sex": 0, "level": 1.25, "target": 1},
            {"age": 51, "sex": 0, "level": 1.5, "target": 1},
        ]
    )
    assert drops(rows, TOY_TRAIN, request) == (None, Drop.RANGE, Drop.CELL, Drop.QUOTA)


def test_check_holds_a_toy_baseline_to_the_toy_contract():
    request = Request(quota={SHORT[0]: 1}, frame=TOY)
    added = toy_added([{"age": 50, "sex": 0, "level": 1.25, "target": 1}])
    weights = pd.Series(1.0, index=TOY_TRAIN.index, name="weight")
    check(Baseline(added, weights, {}), TOY_TRAIN, request)
    with pytest.raises(OutputContractError, match="level has extra decimals"):
        check(Baseline(added.assign(level=1.255), weights, {}), TOY_TRAIN, request)
    with pytest.raises(OutputContractError, match="columns are"):
        check(Baseline(no_rows(), weights, {}), TOY_TRAIN, Request(frame=TOY))
    assert no_rows(TOY).columns.tolist() == ["age", "sex", "level", "target"]


def test_smote_fills_a_toy_request():
    request = Request(quota={SHORT[0]: 3, SHORT[1]: 2}, frame=TOY, seed=0)
    baseline = augment(TOY_TRAIN, Method.SMOTE, request)
    assert baseline.added.columns.tolist() == ["age", "sex", "level", "target"]
    assert len(baseline.added) == 5
    real = training_set(TOY_TRAIN, baseline, frame=TOY)
    assert real.columns.tolist() == ["age", "sex", "level", "target", "weight"]
    hundredths = (baseline.added["level"] * 100).round().astype(int)
    assert (hundredths % 10 != 0).any(), "SMOTE kept one decimal, not the toy frame's two"


@pytest.mark.parametrize("method", [Method.CTGAN, Method.TVAE])
def test_a_synthesizer_fills_a_toy_request(method):
    request = Request(quota={SHORT[0]: 2}, frame=TOY, seed=0)
    baseline = augment(TOY_TRAIN, method, request, methods=FAST_METHODS)
    assert baseline.added.columns.tolist() == ["age", "sex", "level", "target"]
    assert baseline.added["age"].between(45, 69).all()
    assert baseline.added["target"].eq(1).all()


def test_the_sdv_metadata_reads_the_frame():
    assert metadata(TOY) == {
        "tables": {
            "toy": {
                "columns": {
                    "age": {"sdtype": "numerical", "computer_representation": "Int64"},
                    "sex": {"sdtype": "categorical"},
                    "level": {"sdtype": "numerical", "computer_representation": "Float"},
                    "target": {"sdtype": "categorical"},
                }
            }
        }
    }
    assert metadata(UCI_FRAME) is METADATA
    tables: Any = fidelity_module.metadata(TOY)["tables"]
    assert list(tables) == ["toy"]


def test_the_classifier_reads_the_frame():
    transformers = build(TOY).named_steps["features"].transformers
    assert [(name, columns) for name, _, columns in transformers] == [
        ("measured", ["age", "level"]),
        ("coded", ["sex"]),
    ]
    probability = probabilities(TOY_TRAIN.assign(weight=1.0), TOY_PARTS.test, frame=TOY)
    assert probability.between(0, 1).all()
    assert probability.index.equals(TOY_PARTS.test.index)


def test_fidelity_reads_the_toy_table():
    scores = fidelity(TOY_TRAIN, TOY_PARTS.test, frame=TOY)
    assert set(scores) == {Measure.QUALITY, Measure.COLUMN_SHAPES, Measure.COLUMN_PAIR_TRENDS}
    assert all(0 <= value <= 1 for value in scores.values())


def test_copies_and_distances_read_the_toy_columns():
    test = TOY_PARTS.test
    rows = test.head(2).assign(extra=1)
    assert copies(rows, test, frame=TOY) == 2
    assert decontaminated(test, [rows], frame=TOY).index.equals(test.index[2:])


def test_distances_read_the_toy_metadata():
    train = toy_added(
        [
            {"age": 30, "sex": 0, "level": 0.5, "target": 0},
            {"age": 90, "sex": 1, "level": 4.5, "target": 1},
        ]
    )
    rows = toy_added(
        [
            {"age": 30, "sex": 0, "level": 4.5, "target": 0},
            {"age": 60, "sex": 1, "level": 4.5, "target": 1},
        ]
    )
    # Nearest rows: a level gap of 4 over a range of 4, then an age gap of 30 over 60, each
    # averaged over the 4 columns, so 0.25 and 0.125.
    near = distances(rows, train, frame=TOY)
    assert near == pytest.approx({Measure.DCR_P5: 0.13125, Measure.DCR_MEDIAN: 0.1875})


def test_scopes_and_the_held_out_ratio_read_the_toy_edges():
    test = TOY_PARTS.test
    covered = in_scope(test, Scope.FLAGGED, SHORT, edges=TOY.edges)
    assert covered.tolist() == test["age"].between(45, 69).tolist()
    probability = pd.Series(0.6, index=test.index)
    scores = utility(test, probability, SHORT, edges=TOY.edges)
    assert not math.isnan(scores[Measure.RECALL, Scope.FLAGGED])
    # Toy cells of 25 hold out 5 of them, 20 kept; under UCI's edge the ratio would be 9/26.
    assert held_out_ratio(TOY_PARTS, edges=TOY.edges) == pytest.approx(0.25)


def test_the_experiment_requests_read_the_toy_frame():
    assert flagged(TOY_TRAIN, TOY_EXPERIMENT, edges=TOY.edges) == SHORT
    gap = gap_request(TOY_TRAIN, TOY_EXPERIMENT, seed=0, frame=TOY)
    assert gap.frame is TOY
    assert list(gap.quota) == list(SHORT)
    replica = replica_request(TOY_TRAIN, seed=0, frame=TOY)
    assert replica.frame is TOY
    assert replica.rows == len(TOY_TRAIN)


def test_a_toy_experiment_runs_end_to_end():
    arms = {name: ARMS[name] for name in TOY_EXPERIMENT.arms}
    scores = run(toy_records(), TOY_EXPERIMENT, arms, frame=TOY)
    added = [
        s for s in scores if (s.arm, s.source, s.measure) == ("smote", Source.ADDED, Measure.ROWS)
    ]
    assert [s.value for s in added] == [30.0, 30.0]
    assert {s.holdout for s in scores} == {0, 1}
    ratios = [s.value for s in scores if s.measure is Measure.HELD_OUT_RATIO]
    assert ratios == pytest.approx([0.25, 0.25])
    flagged_recall = [
        s for s in scores if (s.arm, s.measure, s.scope) == ("smote", Measure.RECALL, Scope.FLAGGED)
    ]
    assert {s.source for s in flagged_recall} == {Source.AUGMENTED, Source.REPLICA}
    assert not any(math.isnan(s.value) for s in flagged_recall)


def test_a_rehearsal_refuses_another_frame():
    with pytest.raises(ValueError, match="UCI records only, not frame toy"):
        CannedSend(GRADED, SMALL, SETTINGS, frame=TOY)


def test_a_rehearsal_refuses_a_call_for_another_frame():
    send = CannedSend(GRADED, SMALL, SETTINGS)
    with pytest.raises(ValueError, match="toy"):
        send(Call(SETTINGS, 0, 1, "system", "user", frame=TOY))
