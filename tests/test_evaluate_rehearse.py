"""`score --rehearse`: every BKO call answered with canned records, so a run's rules run for free.

No test reaches the API: a rehearsal never calls `Context.send`, and each test's send refuses.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from bko.baselines.records import within_split
from bko.datasets.cells import SEXES, cell_keys
from bko.datasets.split import split
from bko.datasets.uci import FRAME_COLUMNS, NUMERIC, UCI_FRAME
from bko.evaluate import rehearse
from bko.evaluate.__main__ import main
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import RUNS, NamedRun
from bko.evaluate.holdout_candidates import read
from bko.evaluate.rehearse import (
    INPUT_PER_RECORD,
    OUTPUT_PER_RECORD,
    PROMPT,
    CannedSend,
    safe_ranges,
)
from bko.generate.errors import GenerationError
from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import LEVELS, SETTINGS, Call, Level, reaches
from bko.generate.prompt import COHERENCE, render
from bko.validate.checks import validate
from tests._evaluate_bko_fixtures import NARROW, candidates_for, context, score_argv
from tests._evaluate_fixtures import GRADED, GRADED_DATA, SMALL, uci_data
from tests._frame_fixtures import TOY, toy_records
from tests._generate_fixtures import GUIDANCE, OTHER_GUIDANCE, ScriptedSend


SEXES_BY_LABEL = {label: code for code, label in SEXES.items()}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("level", LEVELS)
def test_the_pattern_reads_the_user_turn_at_every_level(level):
    _, user = render(GUIDANCE, 7, level, frame=UCI_FRAME)
    found = PROMPT.match(user)
    assert found is not None
    low, high = GUIDANCE.interval["age"]
    assert (int(found["rows"]), int(found["sex"]), int(found["label"])) == (7, 0, 1)
    assert (float(found["low"]), float(found["high"])) == (low, high)


def test_the_prompt_regex_reads_the_bounded_first_sentence():
    train = split(GRADED, seed=SMALL.holdout_seeds[0]).train
    for cell in sorted(_cells(GRADED)):
        interval = intervals(cell, train, frame=UCI_FRAME)
        sex = SEXES_BY_LABEL[cell[1]]
        guidance = Guidance(cell=cell, sex=sex, label=cell[2], records=3, interval=interval)
        _, user = render(guidance, 4, SETTINGS.guidance, frame=UCI_FRAME)
        found = PROMPT.match(user)
        assert found is not None, cell
        assert (float(found["low"]), float(found["high"])) == interval["age"]
        assert interval["age"][0] >= train["age"].min()
        assert interval["age"][1] <= train["age"].max()


def test_the_safe_ranges_lie_inside_every_holdouts_training_range():
    ranges = safe_ranges(GRADED, SMALL.holdout_seeds, frame=UCI_FRAME)
    assert set(ranges) == set(NUMERIC)
    for seed in SMALL.holdout_seeds:
        train = split(GRADED, seed=seed).train
        for name, (low, high) in ranges.items():
            assert train[name].min() <= low <= high <= train[name].max()


def _cells(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys = cell_keys(frame)
    return {(str(a), str(s), int(t)) for a, s, t in keys.itertuples(index=False)}


def test_the_tokens_per_record_are_the_judged_runs():
    # e1-4's call log end line: input and output tokens over records asked (docs/EVIDENCE.md, E1).
    assert (
        round(23287576 / 77487),
        round(16362315 / 77487),
    ) == (INPUT_PER_RECORD, OUTPUT_PER_RECORD)


def test_canned_records_are_valid_in_cell_and_inside_every_holdouts_range():
    send = CannedSend(GRADED_DATA, SMALL, SETTINGS)
    trains = [split(GRADED, seed=seed).train for seed in SMALL.holdout_seeds]
    real = set(GRADED.loc[:, list(FRAME_COLUMNS)].itertuples(index=False, name=None))
    for cell in sorted(_cells(GRADED)):
        sex, label = SEXES_BY_LABEL[cell[1]], cell[2]
        interval = intervals(cell, trains[0], frame=UCI_FRAME)
        guidance = Guidance(cell=cell, sex=sex, label=label, records=3, interval=interval)
        call = Call(
            SETTINGS,
            sex,
            label,
            *render(guidance, 20, SETTINGS.guidance, frame=UCI_FRAME),
            frame=UCI_FRAME,
        )
        answer = send(call)
        assert answer.model == SETTINGS.model
        assert answer.usage.input_tokens == INPUT_PER_RECORD * 20
        assert answer.usage.output_tokens == OUTPUT_PER_RECORD * 20
        records = json.loads(answer.text or "")["records"]
        assert len(records) == 20
        checked = validate(records, frame=UCI_FRAME)
        assert checked.passed == 20
        frame = pd.DataFrame(records, columns=list(FRAME_COLUMNS))
        assert _cells(frame) == {cell}
        for train in trains:
            assert within_split(frame, train).all(axis=None)
        assert not real & set(frame.itertuples(index=False, name=None))


def test_a_call_whose_user_turn_names_another_cell_is_refused():
    send = CannedSend(GRADED_DATA, SMALL, SETTINGS)
    _, user = render(GUIDANCE, 3, SETTINGS.guidance, frame=UCI_FRAME)
    with pytest.raises(GenerationError, match="not the call's"):
        send(Call(SETTINGS, 1, 1, "system", user, frame=UCI_FRAME))


def test_a_rehearsal_runs_with_no_request_and_its_log_says_so(tmp_path):
    ctx, _ = context(ScriptedSend())
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=out) == 0
    lines = _lines(log)
    assert lines[0]["rehearsal"] is True
    assert lines[-1]["rehearsal"] is True
    assert lines[-1]["status"] == "complete"
    assert lines[-1]["calls"] > 0
    assert lines[-1]["live_usd"] > 0
    assert candidates_for(log).exists()
    assert out.getvalue().startswith("holdout,")


def test_a_rehearsal_says_so_before_any_other_run_line(tmp_path, caplog):
    ctx, _ = context(ScriptedSend())
    caplog.set_level(logging.INFO)
    log = tmp_path / "calls.jsonl"
    assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=io.StringIO()) == 0
    first = caplog.records[0]
    assert first.levelno == logging.WARNING
    assert first.getMessage() == (
        "rehearsal canned replies: spend, projection and stop lines are wiring checks, "
        "not paid figures"
    )
    assert sum(m == first.getMessage() for m in caplog.messages) == 1


@pytest.mark.parametrize("level", LEVELS)
def test_a_guidance_runs_calls_are_made_at_its_level(tmp_path, level):
    name = f"g3-{level}"
    ctx, _ = context(ScriptedSend())
    ctx = dataclasses.replace(ctx, runs={name: NamedRun(NARROW, RUNS[name].settings)})
    log = tmp_path / "calls.jsonl"
    argv = ["score", name, *score_argv(log, tmp_path, "--rehearse")[1:]]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    lines = _lines(log)
    assert lines[0]["settings"]["guidance"] == level
    requests = [json.dumps(line["params"]) for line in lines if line["kind"] == "request"]
    assert requests
    coherence = json.dumps(COHERENCE)[1:-1]
    assert all((coherence in params) is reaches(level, Level.COHERENT) for params in requests)


def test_rehearse_needs_a_bko_arm(tmp_path, caplog):
    ctx, _ = context(ScriptedSend())
    unpaid = dataclasses.replace(
        NARROW, arms=("none", "smote"), candidate="smote", comparators=("none",)
    )
    ctx = dataclasses.replace(ctx, experiment=unpaid)
    argv = ["score", "--rehearse", "--data-dir", str(tmp_path / "data")]
    with caplog.at_level(logging.ERROR):
        assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "--rehearse needs a BKO arm" in caplog.text


def test_a_rehearsal_billed_past_the_projection_stops(tmp_path, caplog):
    ctx, _ = context(ScriptedSend())
    costly = functools.partial(CannedSend, per_record=(10**5, 10**5))
    ctx = dataclasses.replace(
        ctx,
        experiment=dataclasses.replace(NARROW, projection_after=1, spend_ceiling=3.0),
        rehearse=costly,
    )
    log = tmp_path / "calls.jsonl"
    with caplog.at_level(logging.INFO):
        assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=io.StringIO()) == 2
    assert "stop rule=projection" in caplog.text
    assert _lines(log)[-1]["status"] == "failed"


def test_a_rehearsals_log_replays_but_never_resumes(tmp_path, caplog):
    ctx, _ = context(ScriptedSend())
    log = tmp_path / "calls.jsonl"
    first = io.StringIO()
    assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=first) == 0
    resumed = tmp_path / "resumed.jsonl"
    with caplog.at_level(logging.ERROR):
        code = main(score_argv(resumed, tmp_path, "--resume", str(log)), context=ctx)
    assert code == 3
    assert "rehearsal" in caplog.text
    assert not resumed.exists()
    replayed = io.StringIO()
    again = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(log))
    assert main(again, context=ctx, out=replayed) == 0
    assert replayed.getvalue() == first.getvalue()


def test_a_replay_of_a_rehearsal_keeps_the_mark(tmp_path):
    ctx, _ = context(ScriptedSend())
    log = tmp_path / "calls.jsonl"
    assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=io.StringIO()) == 0
    replay = tmp_path / "replay.jsonl"
    argv = score_argv(replay, tmp_path, "--replay", str(log))
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    lines = _lines(replay)
    assert lines[0]["rehearsal"] is True
    assert lines[0]["replay"] is not None
    assert lines[-1]["rehearsal"] is True


def test_the_safe_range_is_the_highest_minimum_to_the_lowest_maximum():
    # Holdout 2's training split lacks the lowest chol, holdout 3's the highest.
    mins = [split(GRADED, seed=seed).train["chol"].min() for seed in (2, 3)]
    maxes = [split(GRADED, seed=seed).train["chol"].max() for seed in (2, 3)]
    assert len(set(mins)) == 2
    assert len(set(maxes)) == 2
    assert safe_ranges(GRADED, (2, 3), frame=UCI_FRAME)["chol"] == (max(mins), min(maxes))


def _one(send: CannedSend) -> dict:
    call = Call(
        SETTINGS,
        1,
        0,
        *render(OTHER_GUIDANCE, 1, SETTINGS.guidance, frame=UCI_FRAME),
        frame=UCI_FRAME,
    )
    (record,) = json.loads(send(call).text or "")["records"]
    return record


def _with(record: dict) -> pd.DataFrame:
    """`GRADED` with `record` added as a real record under a new id."""
    planted = pd.DataFrame(
        [record], index=[int(GRADED.index.max()) + 1], columns=list(FRAME_COLUMNS)
    )
    planted.index.name = GRADED.index.name
    return pd.concat([GRADED, planted.astype(GRADED.dtypes.to_dict())])


def test_a_canned_record_equal_to_a_real_one_is_skipped(monkeypatch):
    first = _one(CannedSend(GRADED_DATA, SMALL, SETTINGS))
    frame = _with(first)
    assert safe_ranges(frame, SMALL.holdout_seeds, frame=UCI_FRAME) == safe_ranges(
        GRADED, SMALL.holdout_seeds, frame=UCI_FRAME
    )
    send = CannedSend(uci_data(frame), SMALL, SETTINGS)
    assert _one(send) != first
    assert send.records == 2
    monkeypatch.setattr(rehearse, "SKIPS", 1)
    with pytest.raises(EvaluationError, match="equal real records"):
        _one(CannedSend(uci_data(frame), SMALL, SETTINGS))


def test_every_candidate_of_a_rehearsal_is_kept(tmp_path):
    ctx, _ = context(ScriptedSend())
    log = tmp_path / "calls.jsonl"
    assert main(score_argv(log, tmp_path, "--rehearse"), context=ctx, out=io.StringIO()) == 0
    lines = read(candidates_for(log).read_bytes()).candidates
    assert lines
    assert {line.outcome for line in lines} == {"kept"}


def test_rehearsal_safe_ranges_use_the_frames_edges():
    records = toy_records()
    seeds = (0, 1)
    found = safe_ranges(records, seeds, frame=TOY)
    trains = [split(records, seed=seed, edges=TOY.edges).train for seed in seeds]
    assert set(found) == set(TOY.numeric)
    for name in TOY.numeric:
        low = max(float(train[name].min()) for train in trains)
        high = min(float(train[name].max()) for train in trains)
        assert found[name] == (low, high)
