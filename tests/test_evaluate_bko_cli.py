"""`python -m bko.evaluate` with a BKO arm: one call log, validity judged, API failures exit 2.

Every call goes to a `ScriptedSend`; no test reaches the API.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import bko.evaluate.guided as guided_module
from bko.bkb.errors import QueryError
from bko.evaluate.__main__ import Context, main
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS
from bko.generate.spend import cost
from tests._evaluate_bko_fixtures import (
    BAD,
    NARROW,
    REPLICA_QUOTA,
    REPLICAS,
    THREE,
    candidates_for,
    context,
    replica_script,
    score_argv,
    scripted,
    three_arm_script,
)
from tests._evaluate_fixtures import FAST_ARMS
from tests._fusion_fixtures import young
from tests._generate_fixtures import ScriptedSend, reply


def test_score_runs_a_bko_arm_and_compare_judges_its_validity(tmp_path, caplog):
    send = scripted()
    ctx, _ = context(send)
    out = io.StringIO()
    log = tmp_path / "calls.jsonl"
    argv = score_argv(log, tmp_path)
    caplog.set_level(logging.INFO)
    assert main(argv, context=ctx, out=out) == 0
    send.assert_consumed()
    frame = pd.read_csv(io.StringIO(out.getvalue()))
    rows = frame[(frame["arm"] == "bko") & (frame["source"] == "candidates")]
    generated = rows[rows["measure"] == "generated"]
    assert sorted(generated["holdout"]) == sorted(NARROW.holdout_seeds)
    assert set(generated["value"]) == {28.0}
    scores = tmp_path / "scores.csv"
    scores.write_text(out.getvalue(), encoding="utf-8")
    assert (
        main(["compare", str(scores)], context=Context(experiment=NARROW), out=io.StringIO()) == 0
    )
    assert "criterion=validity verdict=met arm=bko generated=56 passed=56" in caplog.text
    assert "mean arm=bko" in caplog.text


def test_score_needs_a_log_for_bko_arms(tmp_path):
    send = scripted()
    ctx, _ = context(send)
    out = io.StringIO()
    assert main(["score", "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 2
    assert out.getvalue() == ""
    assert send.calls == []


def test_score_refuses_an_existing_log_before_any_download(tmp_path):
    send = scripted()
    ctx, opener = context(send)
    log = tmp_path / "calls.jsonl"
    log.write_text("", encoding="utf-8")
    argv = score_argv(log, tmp_path)
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert opener.calls == []
    assert send.calls == []


def test_a_call_with_no_response_exits_two_and_logs_the_sums(tmp_path, caplog):
    send = ScriptedSend(GenerationError("the API could not be reached: APIConnectionError"))
    ctx, _ = context(send)
    out = io.StringIO()
    log = tmp_path / "calls.jsonl"
    argv = score_argv(log, tmp_path)
    caplog.set_level(logging.INFO)
    assert main(argv, context=ctx, out=out) == 2
    assert out.getvalue() == ""
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "failed"
    assert "validity arm=bko" in caplog.text


def test_a_stand_in_under_a_bko_name_builds_no_arm(tmp_path):
    send = scripted()
    ctx, _ = context(send, arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]})
    out = io.StringIO()
    assert main(["score", "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    assert out.getvalue() != ""
    assert send.calls == []


def paid_run(tmp_path, caplog) -> tuple[Context, Path, str]:
    """Run `score` over `scripted()` replies; return its context, call log and CSV."""
    ctx, _ = context(scripted())
    out = io.StringIO()
    log = tmp_path / "calls.jsonl"
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 0
    return ctx, log, out.getvalue()


def _validity(caplog) -> list[str]:
    return [m for m in caplog.messages if m.startswith("validity ")]


def _edited(log: Path, target: Path, edit) -> Path:
    """Write `log`'s JSON lines to `target` after `edit` rewrote the list."""
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    edit(lines)
    target.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return target


def test_a_replayed_run_writes_the_live_runs_bytes(tmp_path, caplog):
    ctx, log, text = paid_run(tmp_path, caplog)
    live = _validity(caplog)
    caplog.clear()
    again = io.StringIO()
    replaying = dataclasses.replace(ctx, send=ScriptedSend)
    argv = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(log))
    assert main(argv, context=replaying, out=again) == 0
    assert again.getvalue() == text
    first = json.loads((tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["replay"] == hashlib.sha256(log.read_bytes()).hexdigest()
    end = json.loads((tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert end["usd"] > 0
    assert end["live_usd"] == 0
    assert _validity(caplog) == live


def test_a_replay_with_calls_left_is_refused(tmp_path, caplog):
    ctx, log, _ = paid_run(tmp_path, caplog)

    def extend(lines: list[dict]) -> None:
        for kind in ("request", "reply"):
            last = next(line for line in reversed(lines) if line["kind"] == kind)
            lines.append({**last, "call": 5})

    moved = _edited(log, tmp_path / "more.jsonl", extend)
    out = io.StringIO()
    argv = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
    assert main(argv, context=ctx, out=out) == 3
    assert "1 logged calls were not replayed" in caplog.text
    assert out.getvalue() == ""


def test_a_replay_whose_request_moved_is_refused(tmp_path, caplog):
    ctx, log, _ = paid_run(tmp_path, caplog)

    def move(lines: list[dict]) -> None:
        first = next(line for line in lines if line["kind"] == "request")
        first["params"]["max_tokens"] = 1

    moved = _edited(log, tmp_path / "moved.jsonl", move)
    out = io.StringIO()
    argv = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
    assert main(argv, context=ctx, out=out) == 3
    assert "differs from logged call 1" in caplog.text


def test_replay_needs_bko_arms(tmp_path, caplog):
    ctx, log, _ = paid_run(tmp_path, caplog)
    stand_ins = dataclasses.replace(
        ctx, arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]}, send=ScriptedSend
    )
    out = io.StringIO()
    argv = ["score", "--replay", str(log), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=stand_ins, out=out) == 2
    assert out.getvalue() == ""


def test_log_needs_a_bko_arm_to_build(tmp_path, caplog):
    send = scripted()
    ctx, opener = context(send, arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]})
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 2
    assert "--log needs a BKO arm" in caplog.text
    assert out.getvalue() == ""
    assert not log.exists()
    assert opener.calls == []


def test_an_unreadable_validity_keeps_the_criteria_table(tmp_path, caplog):
    _, _, text = paid_run(tmp_path, caplog)
    frame = pd.read_csv(io.StringIO(text))
    holdout = NARROW.holdout_seeds[0]
    gone = (frame["arm"] == "bko") & (frame["source"] == "candidates")
    scores = tmp_path / "scores.csv"
    frame[~(gone & (frame["holdout"] == holdout))].to_csv(scores, index=False)
    caplog.clear()
    out = io.StringIO()
    assert main(["compare", str(scores)], context=Context(experiment=NARROW), out=out) == 2
    assert "criterion=primary verdict=" in caplog.text
    assert "the generated counts of 'bko' in its gap fill cover 1 holdouts" in caplog.text
    table = pd.read_csv(io.StringIO(out.getvalue()))
    assert list(dict.fromkeys(table["criterion"])) == ["primary", "secondary", "privacy"]


def _end(log: Path) -> dict:
    return json.loads(log.read_text(encoding="utf-8").splitlines()[-1])


def test_three_bko_arms_share_one_log_and_replay_byte_for_byte(tmp_path, caplog):
    send = three_arm_script()
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=THREE)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 0
    send.assert_consumed()
    live = _validity(caplog)
    assert live == [
        "validity arm=bko generated=56 passed=56 dropped_range=0 dropped_cell=0 dropped_quota=0",
        (
            "validity arm=bko-population generated=58 passed=58 dropped_range=2 dropped_cell=0 "
            "dropped_quota=0"
        ),
        (
            "validity arm=bko-frequency generated=60 passed=60 dropped_range=0 dropped_cell=0 "
            "dropped_quota=4"
        ),
    ]
    frame = pd.read_csv(io.StringIO(out.getvalue()))
    candidates = frame[frame["source"] == "candidates"]
    sums = candidates.groupby(["arm", "measure"])["value"].sum()
    assert sums["bko"]["generated"] == 56
    assert sums["bko-population"]["dropped-range"] == 2
    assert sums["bko-frequency"]["dropped-quota"] == 4
    assert sums["bko"]["dropped-range"] == sums["bko"]["dropped-quota"] == 0
    caplog.clear()
    again = io.StringIO()
    replaying = dataclasses.replace(ctx, send=ScriptedSend)
    replayed = tmp_path / "replay.jsonl"
    argv = score_argv(replayed, tmp_path, "--replay", str(log))
    assert main(argv, context=replaying, out=again) == 0
    assert again.getvalue() == out.getvalue()
    assert _validity(caplog) == live
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    assert _end(replayed)["replay"] == digest
    assert _end(log)["replay"] is None
    assert any(
        m.startswith("run status=complete") and f"replay={digest}" in m for m in caplog.messages
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(out.getvalue(), encoding="utf-8")
    compare_three(scores, caplog)


def compare_three(scores: Path, caplog) -> None:
    """`compare` judges only its candidate's validity and reports the other BKO arms'."""
    caplog.clear()
    judged = Context(experiment=THREE)
    assert main(["compare", str(scores)], context=judged, out=io.StringIO()) == 0
    verdicts = [m for m in caplog.messages if m.startswith("criterion=validity ")]
    assert len(verdicts) == 1
    assert verdicts[0].startswith("criterion=validity verdict=met arm=bko ")
    reports = [m.split()[1] for m in caplog.messages if m.startswith("validity ")]
    assert reports == ["arm=bko-population", "arm=bko-frequency"]
    fills = [m.split()[1:3] for m in caplog.messages if m.startswith("candidates ")]
    assert fills == [
        [f"arm={arm}", "fill=gap"] for arm in ("bko", "bko-population", "bko-frequency")
    ]
    caplog.clear()
    argv = ["compare", str(scores), "--candidate", "bko-population"]
    assert main(argv, context=judged, out=io.StringIO()) == 0
    verdicts = [m for m in caplog.messages if m.startswith("criterion=validity ")]
    assert len(verdicts) == 1
    assert verdicts[0].startswith("criterion=validity verdict=met arm=bko-population ")
    reports = [m.split()[1] for m in caplog.messages if m.startswith("validity ")]
    assert reports == ["arm=bko", "arm=bko-frequency"]
    fills = [m.split()[1:3] for m in caplog.messages if m.startswith("candidates ")]
    assert fills == [
        [f"arm={arm}", "fill=gap"] for arm in ("bko-population", "bko", "bko-frequency")
    ]


def test_a_refusal_retry_and_top_up_replay_byte_for_byte(tmp_path, caplog):
    dropped = (reply([BAD] * 12), reply([BAD] * 16))
    refused = dropped * 3
    retried = (reply([young(0)] * 12), reply([*[young(1)] * 15, BAD]), reply([young(1)]))
    second = (reply([young(0)] * 12), reply([young(1)] * 16))
    send = ScriptedSend(*refused, *retried, *second)
    ctx, _ = context(send)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 0
    send.assert_consumed()
    assert any(m.startswith("retry holdout=") and "arm=bko attempt=1" in m for m in caplog.messages)
    asked = [
        line["asked"]
        for line in map(json.loads, log.read_text(encoding="utf-8").splitlines())
        if line["kind"] == "request"
    ]
    assert asked[6:9] == [12, 16, 1]
    live = _validity(caplog)
    assert live == [
        "validity arm=bko generated=141 passed=141 dropped_range=85 dropped_cell=0 dropped_quota=0"
    ]
    caplog.clear()
    again = io.StringIO()
    replaying = dataclasses.replace(ctx, send=ScriptedSend)
    argv = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(log))
    assert main(argv, context=replaying, out=again) == 0
    assert again.getvalue() == out.getvalue()
    assert _validity(caplog) == live


def test_the_ceiling_stops_the_run_unretried_across_arms(tmp_path, caplog):
    send = three_arm_script()
    ctx, _ = context(send)
    spent = cost(reply([]).usage, SETTINGS.model)
    ceiling = spent + spent + spent
    ctx = dataclasses.replace(ctx, experiment=dataclasses.replace(THREE, spend_ceiling=ceiling))
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 2
    assert out.getvalue() == ""
    assert len(send.calls) == 3
    assert _end(log)["status"] == "failed"
    assert _end(log)["calls"] == 3
    assert "reached its ceiling" in caplog.text
    assert not any(m.startswith("retry ") for m in caplog.messages)
    assert [m.split()[1] for m in _validity(caplog)] == [
        "arm=bko",
        "arm=bko-population",
        "arm=bko-frequency",
    ]


def test_the_projection_stops_a_live_run_between_holdouts(tmp_path, caplog):
    send = scripted()
    ctx, _ = context(send)
    spent = cost(reply([]).usage, SETTINGS.model)
    # One holdout spends at least two calls' worth; scaled to two holdouts that reaches four.
    rules = dataclasses.replace(NARROW, projection_after=1, spend_ceiling=3 * spent)
    ctx = dataclasses.replace(ctx, experiment=rules)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 2
    assert out.getvalue() == ""
    assert len(send.calls) == 2
    assert _end(log)["status"] == "failed"
    assert "stop rule=projection" in caplog.text
    assert "reached its ceiling" not in caplog.text


def test_a_failure_mid_run_keeps_the_calls_before_it_in_the_sums(tmp_path, caplog):
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    failure = GenerationError("the API could not be reached: APIConnectionError")
    send = ScriptedSend(*holdout, holdout[0], failure)
    ctx, _ = context(send)
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(tmp_path / "calls.jsonl", tmp_path), context=ctx, out=out) == 2
    assert out.getvalue() == ""
    guided = [m for m in caplog.messages if m.startswith("guided arm=bko ")]
    assert "outcome=returned calls=1-2 " in guided[0]
    assert "outcome=failed calls=3-4 " in guided[1]
    assert _validity(caplog) == [
        "validity arm=bko generated=40 passed=40 dropped_range=0 dropped_cell=0 dropped_quota=0"
    ]


def test_a_failed_gap_query_exits_two_unretried(tmp_path, caplog, monkeypatch):
    queried: list[object] = []

    def refuse(*args: object, **kwargs: object) -> object:
        queried.append(args)
        raise QueryError("a cell assumes a concept no fragment holds")

    monkeypatch.setattr(guided_module, "gaps", refuse)
    send = scripted()
    ctx, _ = context(send)
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(tmp_path / "calls.jsonl", tmp_path), context=ctx, out=out) == 2
    assert out.getvalue() == ""
    assert len(queried) == 1
    assert send.calls == []
    assert "no fragment holds" in caplog.text
    assert _validity(caplog) == [
        "validity arm=bko generated=0 passed=0 dropped_range=0 dropped_cell=0 dropped_quota=0"
    ]


def test_a_replay_whose_second_request_moved_is_refused(tmp_path, caplog):
    ctx, log, _ = paid_run(tmp_path, caplog)

    def move(lines: list[dict]) -> None:
        second = [line for line in lines if line["kind"] == "request"][1]
        second["params"]["system"] += " "

    moved = _edited(log, tmp_path / "moved.jsonl", move)
    out = io.StringIO()
    argv = score_argv(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
    assert main(argv, context=ctx, out=out) == 3
    assert "differs from logged call 2" in caplog.text
    assert out.getvalue() == ""


def test_every_bko_arm_fills_a_replica_counted_apart_and_replayed(tmp_path, caplog):
    send = replica_script()
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=REPLICAS)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(score_argv(log, tmp_path), context=ctx, out=out) == 0
    send.assert_consumed()
    lines = map(json.loads, log.read_text(encoding="utf-8").splitlines())
    asked = [line["asked"] for line in lines if line["kind"] == "request"]
    assert asked == ([12, 16] * 3 + [rows for *_, rows in REPLICA_QUOTA] * 3) * 2
    live = _validity(caplog)
    assert live == [
        "validity arm=bko generated=322 passed=320 dropped_range=0 dropped_cell=0 dropped_quota=0",
        (
            "validity arm=bko-population generated=328 passed=328 dropped_range=8 dropped_cell=0 "
            "dropped_quota=0"
        ),
        (
            "validity arm=bko-frequency generated=326 passed=326 dropped_range=0 dropped_cell=0 "
            "dropped_quota=6"
        ),
    ]
    frame = pd.read_csv(io.StringIO(out.getvalue()))
    assert set(frame[frame["source"] == "replica"]["arm"]) == set(REPLICAS.replica_arms)
    caplog.clear()
    again = io.StringIO()
    replayed = tmp_path / "replay.jsonl"
    argv = score_argv(replayed, tmp_path, "--replay", str(log))
    assert main(argv, context=dataclasses.replace(ctx, send=ScriptedSend), out=again) == 0
    assert again.getvalue() == out.getvalue()
    assert _validity(caplog) == live
    assert _end(replayed)["replay"] == hashlib.sha256(log.read_bytes()).hexdigest()
    scores = tmp_path / "scores.csv"
    scores.write_text(out.getvalue(), encoding="utf-8")
    caplog.clear()
    judged = Context(experiment=REPLICAS)
    assert main(["compare", str(scores)], context=judged, out=io.StringIO()) == 0
    assert [m for m in caplog.messages if m.startswith("candidates ")] == [
        (
            "candidates arm=bko fill=gap generated=56 passed=56 dropped_range=0 dropped_cell=0 "
            "dropped_quota=0 rate=1.0000"
        ),
        (
            "candidates arm=bko fill=replica generated=266 passed=264 dropped_range=0 "
            "dropped_cell=0 dropped_quota=0 rate=0.9925"
        ),
        (
            "candidates arm=bko-population fill=gap generated=58 passed=58 dropped_range=2 "
            "dropped_cell=0 dropped_quota=0 rate=1.0000"
        ),
        (
            "candidates arm=bko-population fill=replica generated=270 passed=270 dropped_range=6 "
            "dropped_cell=0 dropped_quota=0 rate=1.0000"
        ),
        (
            "candidates arm=bko-frequency fill=gap generated=60 passed=60 dropped_range=0 "
            "dropped_cell=0 dropped_quota=4 rate=1.0000"
        ),
        (
            "candidates arm=bko-frequency fill=replica generated=266 passed=266 dropped_range=0 "
            "dropped_cell=0 dropped_quota=2 rate=1.0000"
        ),
    ]
    assert "copies arm=bko-frequency added=0 replica=2" in caplog.messages


def stopped_run(tmp_path, caplog) -> tuple[Path, str]:
    """An uninterrupted run's CSV, and the log of the same run stopped at its third call."""
    _, _, text = paid_run(tmp_path, caplog)
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    lost = GenerationError("the API could not be reached: APIConnectionError")
    ctx, _ = context(ScriptedSend(*holdout, lost))
    stopped = tmp_path / "stopped.jsonl"
    assert main(score_argv(stopped, tmp_path), context=ctx, out=io.StringIO()) == 2
    return stopped, text


def resumed_run(tmp_path, caplog) -> tuple[Path, Path, str]:
    """The stopped run's log, the log of its resumption, and the uninterrupted run's CSV."""
    stopped, text = stopped_run(tmp_path, caplog)
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    send = ScriptedSend(*holdout)
    ctx, _ = context(send)
    out = io.StringIO()
    resumed = tmp_path / "resumed.jsonl"
    assert main(score_argv(resumed, tmp_path, "--resume", str(stopped)), context=ctx, out=out) == 0
    send.assert_consumed()
    assert out.getvalue() == text
    return stopped, resumed, text


def test_a_resumed_run_writes_the_uninterrupted_runs_bytes_and_pays_only_its_live_calls(
    tmp_path, caplog
):
    stopped, resumed, _ = resumed_run(tmp_path, caplog)
    lines = [json.loads(line) for line in resumed.read_text(encoding="utf-8").splitlines()]
    digest = hashlib.sha256(stopped.read_bytes()).hexdigest()
    assert lines[0]["resume"] == {"sha256": digest, "calls": 2}
    assert lines[0]["replay"] is None
    live = sum(line["usd"] for line in lines if line["kind"] == "reply" and line["call"] > 2)
    assert lines[-1]["live_usd"] == pytest.approx(live)
    assert lines[-1]["usd"] > lines[-1]["live_usd"] > 0
    assert f"resume log={stopped} sha256={digest} answered=2" in caplog.text


def test_a_resumed_runs_log_replays_to_the_same_bytes(tmp_path, caplog):
    _, resumed, text = resumed_run(tmp_path, caplog)
    ctx, _ = context(ScriptedSend())
    out = io.StringIO()
    argv = score_argv(tmp_path / "again.jsonl", tmp_path, "--replay", str(resumed))
    assert main(argv, context=ctx, out=out) == 0
    assert out.getvalue() == text


def test_resume_and_replay_together_exit_two(tmp_path):
    ctx, _ = context(ScriptedSend())
    argv = score_argv(tmp_path / "x.jsonl", tmp_path, "--resume", "a", "--replay", "b")
    with pytest.raises(SystemExit) as raised:
        main(argv, context=ctx, out=io.StringIO())
    assert raised.value.code == 2


def test_resume_needs_a_bko_arm(tmp_path, caplog):
    ctx, _ = context(ScriptedSend(), arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]})
    argv = score_argv(tmp_path / "x.jsonl", tmp_path, "--resume", str(tmp_path / "s.jsonl"))
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "--resume needs a BKO arm" in caplog.text


def test_an_unresumable_log_exits_three_before_any_download_or_log(tmp_path, caplog):
    _, log, _ = paid_run(tmp_path, caplog)

    def fail_first(lines: list[dict]) -> None:
        index = next(i for i, line in enumerate(lines) if line["kind"] == "reply")
        call = lines[index]["call"]
        lines[index] = {"kind": "error", "call": call, "error": "lost", "attempts": None}

    broken = _edited(log, tmp_path / "broken.jsonl", fail_first)
    send = ScriptedSend()
    ctx, opener = context(send)
    resumed = tmp_path / "resumed.jsonl"
    argv = ["score", "--resume", str(broken), "--log", str(resumed)]
    argv += ["--candidates", str(candidates_for(resumed))]
    argv += ["--data-dir", str(tmp_path / "fresh")]
    assert main(argv, context=ctx, out=io.StringIO()) == 3
    assert "failed before the run's last call" in caplog.text
    assert opener.calls == []
    assert not resumed.exists()
