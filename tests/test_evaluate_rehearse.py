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
from bko.datasets.uci import FRAME_COLUMNS, NUMERIC
from bko.evaluate.__main__ import main
from bko.evaluate.rehearse import (
    INPUT_PER_RECORD,
    OUTPUT_PER_RECORD,
    PROMPT,
    CannedSend,
    safe_ranges,
)
from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import LEVELS, SETTINGS, Call
from bko.generate.prompt import render
from bko.validate.checks import validate
from tests._evaluate_bko_fixtures import NARROW, candidates_for, context, score_argv
from tests._evaluate_fixtures import GRADED, SMALL
from tests._generate_fixtures import GUIDANCE, ScriptedSend


SEXES_BY_LABEL = {label: code for code, label in SEXES.items()}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("level", LEVELS)
def test_the_pattern_reads_the_user_turn_at_every_level(level):
    _, user = render(GUIDANCE, 7, level)
    found = PROMPT.match(user)
    assert found is not None
    low, high = GUIDANCE.interval["age"]
    assert (int(found["rows"]), int(found["sex"]), int(found["label"])) == (7, 0, 1)
    assert (float(found["low"]), float(found["high"])) == (low, high)


def test_the_safe_ranges_lie_inside_every_holdouts_training_range():
    ranges = safe_ranges(GRADED, SMALL.holdout_seeds)
    assert set(ranges) == set(NUMERIC)
    for seed in SMALL.holdout_seeds:
        train = split(GRADED, seed=seed).train
        for name, (low, high) in ranges.items():
            assert train[name].min() <= low <= high <= train[name].max()


def _cells(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys = cell_keys(frame)
    return {(str(a), str(s), int(t)) for a, s, t in keys.itertuples(index=False)}


def test_canned_records_are_valid_in_cell_and_inside_every_holdouts_range():
    send = CannedSend(GRADED, SMALL, SETTINGS)
    trains = [split(GRADED, seed=seed).train for seed in SMALL.holdout_seeds]
    real = set(GRADED.loc[:, list(FRAME_COLUMNS)].itertuples(index=False, name=None))
    for cell in sorted(_cells(GRADED)):
        sex, label = SEXES_BY_LABEL[cell[1]], cell[2]
        guidance = Guidance(cell=cell, sex=sex, label=label, records=3, interval=intervals(cell))
        call = Call(SETTINGS, sex, label, *render(guidance, 20, SETTINGS.guidance))
        answer = send(call)
        assert answer.model == SETTINGS.model
        assert answer.usage.input_tokens == INPUT_PER_RECORD * 20
        assert answer.usage.output_tokens == OUTPUT_PER_RECORD * 20
        records = json.loads(answer.text or "")["records"]
        assert len(records) == 20
        checked = validate(records)
        assert checked.passed == 20
        frame = pd.DataFrame(records, columns=list(FRAME_COLUMNS))
        assert _cells(frame) == {cell}
        for train in trains:
            assert within_split(frame, train).all(axis=None)
        assert not real & set(frame.itertuples(index=False, name=None))


def test_a_call_whose_user_turn_names_another_cell_is_refused():
    send = CannedSend(GRADED, SMALL, SETTINGS)
    _, user = render(GUIDANCE, 3, SETTINGS.guidance)
    with pytest.raises(ValueError, match="not the call's"):
        send(Call(SETTINGS, 1, 1, "system", user))


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
