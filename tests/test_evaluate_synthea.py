"""E3 over the Synthea frame: the training-only removal, its prompts and a free end-to-end run.

No test reaches the API: the rehearsal answers every call with canned records, and the live send
refuses.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.baselines.records import within_split
from bko.datasets.ablation import removal_order, thinned
from bko.datasets.cells import cell_keys
from bko.datasets.errors import DatasetError
from bko.datasets.split import split
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, frame_records, load_population
from bko.datasets.synthea_run import RUN_RECORD_NAME
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.data import Data, Dataset, digest, load
from bko.evaluate.experiment import E1, Removal
from bko.evaluate.named_runs import E3, RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.run import method_arms
from bko.evaluate.scores import parse
from bko.gaps.detect import Statement
from bko.generate.guidance import Basis, Finding, Guidance, Profile, Relationship, Share, Spread
from bko.generate.messages import SETTINGS, AnySend, Call, Level
from bko.generate.prompt import render
from bko.ontology.frames import (
    SYNTHEA_ADULTS_FROM,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
    SYNTHEA_INDICATORS,
)
from bko.validate.checks import validate
from tests._evaluate_bko_fixtures import candidates_for
from tests._evaluate_fixtures import FAST_METHODS
from tests._ontology_fixtures import RoutingOpener
from tests._synthea_frame_fixtures import (
    FRAME_RUN,
    JAR_PIN,
    RELEASE,
    RELEASE_PIN,
    write_population,
)


UCI_WORDS = ("chol", "thalach", "trestbps", "oldpeak", "angiography")
"""UCI's columns and cohort, none of which a Synthea prompt may name."""

REMOVED = (">=65", "female", 1)

TWO: Final = dataclasses.replace(E3, holdout_seeds=(0, 1))
"""E3 on two holdouts."""


def _records(data_dir: Path) -> pd.DataFrame:
    population = load_population(write_population(data_dir), snapshot_date=date(2026, 1, 1))
    return frame_records(
        population,
        label=ISCHEMIC_HEART_DISEASE,
        indicators=SYNTHEA_INDICATORS,
        adults_from=SYNTHEA_ADULTS_FROM,
    )


def _in_cell(frame: pd.DataFrame) -> pd.Series:
    keys = cell_keys(frame, edges=SYNTHEA_FRAME.edges)
    return (keys["age_band"] == REMOVED[0]) & (keys["sex"] == REMOVED[1]) & (keys["target"] == 1)


def test_the_removal_thins_training_rows_only(tmp_path):
    records = _records(tmp_path)
    parts = split(records, seed=3, edges=SYNTHEA_FRAME.edges)
    ready = prepared(parts, E3, edges=SYNTHEA_FRAME.edges)
    kept, removed = ready.ablated, ready.removed
    pd.testing.assert_frame_equal(kept.test, parts.test)
    assert kept.seed == parts.seed
    selected = parts.train.index[_in_cell(parts.train).to_numpy()].sort_values()
    count = math.ceil(Fraction("0.9") * len(selected))
    drawn = removal_order(selected, 3)[:count]
    assert sorted(removed.index) == sorted(drawn)
    pd.testing.assert_frame_equal(removed, parts.train.loc[removed.index])
    assert sorted(kept.train.index) == sorted(parts.train.index.difference(drawn))
    assert len(selected) - count == int(_in_cell(kept.train).sum())
    again, out = thinned(parts.train, cell=REMOVED, edges=SYNTHEA_FRAME.edges, fraction=0.9, seed=3)
    pd.testing.assert_frame_equal(again, kept.train)
    pd.testing.assert_frame_equal(out, removed)
    whole = prepared(parts, E1, edges=SYNTHEA_FRAME.edges)
    assert whole.ablated is parts
    assert whole.sized is parts
    assert whole.removed.empty


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_thinning_refuses_a_fraction_outside_0_and_1(tmp_path, fraction):
    train = split(_records(tmp_path), seed=0, edges=SYNTHEA_FRAME.edges).train
    with pytest.raises(DatasetError, match="must lie in"):
        thinned(train, cell=REMOVED, edges=SYNTHEA_FRAME.edges, fraction=fraction, seed=0)


def test_thinning_refuses_an_empty_cell(tmp_path):
    train = split(_records(tmp_path), seed=0, edges=SYNTHEA_FRAME.edges).train
    with pytest.raises(DatasetError, match="holds no training row"):
        thinned(train, cell=("<40", "female", 1), edges=(40,), fraction=0.5, seed=0)


def test_thinning_does_not_depend_on_the_rows_order(tmp_path):
    train = split(_records(tmp_path), seed=0, edges=SYNTHEA_FRAME.edges).train
    shuffled = train.sample(frac=1.0, random_state=7)
    assert list(shuffled.index) != list(train.index)
    options = {"cell": REMOVED, "edges": SYNTHEA_FRAME.edges, "fraction": 0.9, "seed": 5}
    removed = thinned(train, **options)[1]
    again = thinned(shuffled, **options)[1]
    assert sorted(again.index) == sorted(removed.index)


def _one_cell(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame({"age": [70] * rows, "sex": [0] * rows, "target": [1] * rows})
    frame.index = pd.RangeIndex(1, rows + 1, name="record_id")
    return frame


@pytest.mark.parametrize(("fraction", "removed"), [(0.29, 29), (0.07, 7)])
def test_thinning_counts_the_fraction_exactly_at_a_float_boundary(fraction, removed):
    assert math.ceil(0.07 * 100) == 8, "the float product overshoots 7"
    kept, out = thinned(_one_cell(100), cell=REMOVED, edges=(65,), fraction=fraction, seed=0)
    assert (len(out), len(kept)) == (removed, 100 - removed)


def test_thinning_at_one_removes_every_row_of_the_cell():
    train = pd.concat([_one_cell(10), _one_cell(3).set_axis(range(11, 14)).assign(target=0)])
    kept, out = thinned(train, cell=REMOVED, edges=(65,), fraction=1.0, seed=0)
    assert sorted(out.index) == list(range(1, 11))
    assert list(kept.index) == [11, 12, 13]


def test_a_removal_fraction_must_lie_in_0_to_1():
    assert Removal(REMOVED, 1.0) == Removal(REMOVED, 1)
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            Removal(REMOVED, fraction)


def _unsendable() -> AnySend:
    msg = "a rehearsal never builds the live send"
    raise AssertionError(msg)


def _context() -> Context:
    return Context(
        open_url=RoutingOpener({RELEASE_PIN.url: RELEASE.encode()}),
        runs={"e3": dataclasses.replace(RUNS["e3"], experiment=TWO)},
        arms=method_arms(FAST_METHODS),
        releases={"MONDO": RELEASE_PIN},
        send=_unsendable,
        synthea=FRAME_RUN,
        synthea_jar=JAR_PIN,
    )


def test_e3_rehearses_end_to_end_on_a_tmp_population(tmp_path):
    data_dir = tmp_path / "data"
    write_population(data_dir)
    log = tmp_path / "calls.jsonl"
    argv = [
        "score",
        "e3",
        "--rehearse",
        "--log",
        str(log),
        "--candidates",
        str(candidates_for(log)),
        "--data-dir",
        str(data_dir),
    ]
    out = io.StringIO()
    assert main(argv, context=_context(), out=out) == 0
    scores = parse(out.getvalue())
    assert {score.arm for score in scores} >= set(E3.arms)
    assert {score.holdout for score in scores} == {0, 1}
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["status"] == "complete"
    requests = [line for line in lines if line["kind"] == "request"]
    assert requests
    for line in requests:
        text = json.dumps(line["params"])
        assert not [word for word in UCI_WORDS if word in text]
        assert "essential_hypertension" in text


def _guidance() -> Guidance:
    hypertension, lipids = SYNTHEA_FIELD_ROWS[1], SYNTHEA_FIELD_ROWS[3]
    statement = Statement("mondo", hypertension.concept, "MONDO:9000001")
    return Guidance(
        cell=REMOVED,
        sex=0,
        label=1,
        records=5,
        interval={"age": (65.0, 95.0)},
        basis=Basis.ABSENT,
        findings=(
            Finding(
                states=(hypertension,),
                relationships=(Relationship(statement, "essential hypertension", "disease"),),
            ),
        ),
        profile=Profile(
            records=120,
            shares=(
                Share(hypertension.concept, hypertension.label, (hypertension,), 0.4),
                Share(lipids.concept, lipids.label, (lipids,), 0.25),
            ),
        ),
        summary={"age": Spread(low=70.0, median=76.0, high=84.0)},
    )


def test_the_synthea_user_turn_at_summary():
    _, user = render(_guidance(), 15, Level.SUMMARY, frame=SYNTHEA_FRAME)
    assert user == (
        "Write exactly 15 records of patients in this group: sex 0 (female), target 1 "
        "(ischemic heart disease), age 65 to 95 years. The training data holds 5 records of "
        "this group.\n"
        "\n"
        "Keep each numeric field within: age 65 to 95 years.\n"
        "\n"
        "Keep each record clinically coherent: its values should fit the patient's disease "
        "status, age and sex together.\n"
        "\n"
        "No record of this group in the training data shows the findings below. At least one "
        "record should show each; otherwise keep each as frequent as is clinically plausible for "
        "this group.\n"
        "- essential_hypertension is 1 (present) (essential hypertension): essential "
        "hypertension is a kind of disease\n"
        "\n"
        "Among comparable patients (the same target, other age and sex groups; 120 records), "
        "these findings occur at these rates:\n"
        "- essential hypertension (essential_hypertension is 1 (present)): 40%\n"
        "- hyperlipidemia (hyperlipidemia is 1 (present)): 25%\n"
        "\n"
        "Among the same patients, numeric fields run as follows (median; middle half):\n"
        "- age: 76; 70 to 84 years"
    )


def test_synthea_prompts_name_no_uci_column():
    system, user = render(_guidance(), 15, Level.SUMMARY, frame=SYNTHEA_FRAME)
    for text in (system, user):
        assert not [word for word in UCI_WORDS if word in text]
    assert system.startswith(
        f"You write synthetic patient records for a study of {SYNTHEA_FRAME.study}. Each record "
        f"is one {SYNTHEA_FRAME.cohort}."
    )
    assert "Every field is a whole number." in system
    for name in SYNTHEA_INDICATORS:
        assert f"- {name}: " in system


def test_canned_synthea_records_are_valid_in_cell_and_equal_no_real_record(tmp_path):
    records = _records(tmp_path)
    data = Data(records, SYNTHEA_FRAME, SYNTHEA_FIELD_ROWS, "synthea", None)
    send = CannedSend(data, TWO, SETTINGS)
    trains = [
        prepared(
            split(records, seed=seed, edges=SYNTHEA_FRAME.edges), E3, edges=(65,)
        ).ablated.train
        for seed in TWO.holdout_seeds
    ]
    real = set(records.loc[:, list(SYNTHEA_FRAME.columns)].itertuples(index=False, name=None))
    guidance = dataclasses.replace(_guidance(), interval={"age": (65.0, 95.0)})
    system, user = render(guidance, 20, SETTINGS.guidance, frame=SYNTHEA_FRAME)
    for _ in range(5):
        reply = send(Call(SETTINGS, 0, 1, system, user, frame=SYNTHEA_FRAME))
        batch = json.loads(reply.text or "")["records"]
        assert len(batch) == 20
        assert validate(batch, frame=SYNTHEA_FRAME).passed == 20
        frame = pd.DataFrame(batch, columns=list(SYNTHEA_FRAME.columns))
        assert _in_cell(frame).all()
        for train in trains:
            assert within_split(frame, train, frame=SYNTHEA_FRAME).all(axis=None)
        assert not real & set(frame.itertuples(index=False, name=None))
    assert send.records == 5 * 20 + 3, "3 drawn records equal a real record and are skipped"


def _loaded(data_dir: Path, caplog) -> Data:
    with caplog.at_level(logging.INFO):
        return load(Dataset.SYNTHEA, data_dir, _context())


def test_loading_synthea_logs_its_data_line(tmp_path, caplog):
    data = _loaded(_written(tmp_path), caplog)
    expected = digest(_records(tmp_path / "again"))
    assert data.digest == expected
    lines = [m for m in caplog.messages if m.startswith("data ")]
    assert lines == [
        f"data dataset=synthea frame=synthea records=310 digest={expected} agreement=none"
    ]


def _written(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    write_population(data_dir)
    return data_dir


def test_loading_synthea_refuses_a_missing_population(tmp_path, caplog):
    with pytest.raises(DatasetError, match="unreadable record"):
        _loaded(tmp_path / "empty", caplog)


def test_loading_synthea_refuses_a_stale_run_record(tmp_path, caplog):
    data_dir = _written(tmp_path)
    (record,) = data_dir.rglob(RUN_RECORD_NAME)
    written = json.loads(record.read_text(encoding="utf-8"))
    written["run"]["seed"] += 1
    record.write_text(json.dumps(written), encoding="utf-8")
    with pytest.raises(DatasetError, match="differs from this run in"):
        _loaded(data_dir, caplog)
