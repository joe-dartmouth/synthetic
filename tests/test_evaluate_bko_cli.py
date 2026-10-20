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
from typing import Final

import pandas as pd

import bko.evaluate.guided as guided_module
from bko.bkb.errors import QueryError
from bko.evaluate.__main__ import Context, main
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS, Reply
from bko.generate.spend import cost
from tests._datasets_fixtures import pin
from tests._evaluate_fixtures import FAST_ARMS, SMALL
from tests._fusion_fixtures import young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, ScriptedSend, reply, uci
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import PLANTED, planted


NARROW: Final = dataclasses.replace(
    SMALL, arms=("none", "bko"), candidate="bko", comparators=("none",), replica_arms=()
)
"""E1 on two holdouts with one baseline and the candidate."""

UCI_E1: Final = uci({(40, 0): "4.0"}, "1.0", {**SIZES, (40, 0, 1): 5})
"""`UCI_X` with five young women with the condition, so every holdout tests one of them.

Each holdout's gap request is then 12 rows of `("<55", "female", 0)` and 16 of
`("<55", "female", 1)`, from 8 and 4 training records."""

PIN: Final = pin(UCI_E1)


def scripted() -> ScriptedSend:
    """Both holdouts' replies: each fills its two cells in one call apiece."""
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    return ScriptedSend(*holdout, *holdout)


def context(send: ScriptedSend, *, arms=None) -> tuple[Context, RoutingOpener]:
    """A context over `UCI_E1` and the ladder releases, with `send` for every call."""
    pins, served = ladder_releases()
    opener = RoutingOpener({**served, PIN.url: UCI_E1})
    built = Context(
        open_url=opener,
        uci=PIN,
        experiment=NARROW,
        arms=arms or {"none": FAST_ARMS["none"]},
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
    )
    return built, opener


def test_score_runs_a_bko_arm_and_compare_judges_its_validity(tmp_path, caplog):
    send = scripted()
    ctx, _ = context(send)
    out = io.StringIO()
    log = tmp_path / "calls.jsonl"
    argv = ["score", "--log", str(log), "--data-dir", str(tmp_path / "data")]
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
    argv = ["score", "--log", str(log), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert opener.calls == []
    assert send.calls == []


def test_a_call_with_no_response_exits_two_and_logs_the_sums(tmp_path, caplog):
    send = ScriptedSend(GenerationError("the API could not be reached: APIConnectionError"))
    ctx, _ = context(send)
    out = io.StringIO()
    log = tmp_path / "calls.jsonl"
    argv = ["score", "--log", str(log), "--data-dir", str(tmp_path / "data")]
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
    assert main(_score(log, tmp_path), context=ctx, out=out) == 0
    return ctx, log, out.getvalue()


def _score(log: Path, tmp_path: Path, *flag: str) -> list[str]:
    return ["score", *flag, "--log", str(log), "--data-dir", str(tmp_path / "data")]


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
    argv = _score(tmp_path / "replay.jsonl", tmp_path, "--replay", str(log))
    assert main(argv, context=replaying, out=again) == 0
    assert again.getvalue() == text
    first = json.loads((tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["replay"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert _validity(caplog) == live


def test_a_replay_with_calls_left_is_refused(tmp_path, caplog):
    ctx, log, _ = paid_run(tmp_path, caplog)

    def extend(lines: list[dict]) -> None:
        for kind in ("request", "reply"):
            last = next(line for line in reversed(lines) if line["kind"] == kind)
            lines.append({**last, "call": 5})

    moved = _edited(log, tmp_path / "more.jsonl", extend)
    out = io.StringIO()
    argv = _score(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
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
    argv = _score(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
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
    assert main(_score(log, tmp_path), context=ctx, out=out) == 2
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


THREE: Final = dataclasses.replace(NARROW, arms=("none", "bko", "bko-population", "bko-frequency"))
"""`NARROW` with all three BKO arms on one call log."""

BAD: Final = young(1, chol=234)
"""A candidate that passes `bko.validate` and is dropped for range: no training record has it."""


def holdout_replies(arm: str) -> tuple[Reply, Reply]:
    """One holdout's two calls for `arm`, asking 12 rows of target 0 then 16 of target 1.

    Each arm's replies differ, so each arm's counts show whose calls they came from:
    `bko-population` gets one record dropped for range per holdout, `bko-frequency` two past the
    quota.
    """
    well: list[object] = [young(0)] * 12
    sick: list[object] = [young(1)] * 16
    if arm == "bko-population":
        sick = [*sick, BAD]
    if arm == "bko-frequency":
        well = [*well, young(0), young(0)]
    return reply(well), reply(sick)


def three_arm_script() -> ScriptedSend:
    """Both holdouts' calls in the order the harness makes them: holdout, then arm, then cell."""
    arms = THREE.arms[1:]
    return ScriptedSend(
        *(r for _ in THREE.holdout_seeds for arm in arms for r in holdout_replies(arm))
    )


def _end(log: Path) -> dict:
    return json.loads(log.read_text(encoding="utf-8").splitlines()[-1])


def test_three_bko_arms_share_one_log_and_replay_byte_for_byte(tmp_path, caplog):
    send = three_arm_script()
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=THREE)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(_score(log, tmp_path), context=ctx, out=out) == 0
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
    argv = _score(replayed, tmp_path, "--replay", str(log))
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
    assert main(_score(log, tmp_path), context=ctx, out=out) == 0
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
    argv = _score(tmp_path / "replay.jsonl", tmp_path, "--replay", str(log))
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
    assert main(_score(log, tmp_path), context=ctx, out=out) == 2
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


def test_a_failure_mid_run_keeps_the_calls_before_it_in_the_sums(tmp_path, caplog):
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    failure = GenerationError("the API could not be reached: APIConnectionError")
    send = ScriptedSend(*holdout, holdout[0], failure)
    ctx, _ = context(send)
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(_score(tmp_path / "calls.jsonl", tmp_path), context=ctx, out=out) == 2
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
    assert main(_score(tmp_path / "calls.jsonl", tmp_path), context=ctx, out=out) == 2
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
    argv = _score(tmp_path / "replay.jsonl", tmp_path, "--replay", str(moved))
    assert main(argv, context=ctx, out=out) == 3
    assert "differs from logged call 2" in caplog.text
    assert out.getvalue() == ""


REPLICAS: Final = dataclasses.replace(THREE, replica_arms=THREE.arms[1:])
"""`THREE` with every BKO arm also filling a replica, as E1 asks of every row-adding arm."""

REPLICA_QUOTA: Final = (
    (40, 0, 0, 8),
    (40, 0, 1, 4),
    (40, 1, 0, 20),
    (40, 1, 1, 20),
    (63, 0, 0, 20),
    (63, 0, 1, 20),
    (63, 1, 0, 20),
    (63, 1, 1, 20),
)
"""`UCI_E1`'s training cells in grid order, as age, sex, target and records: one in five of
`SIZES` held out. Each is one replica call of that many rows."""


def replica_replies(arm: str) -> tuple[Reply, ...]:
    """One holdout's eight replica calls for `arm`, with cp 2 so no record copies a test record.

    Each arm's replica replies differ from its gap replies, so a count under the wrong source
    shows: `bko` gets one candidate that fails validation, `bko-population` three dropped for
    range, and `bko-frequency` one past the quota and one copy of a test record.
    """
    calls: list[list[object]] = []
    for age, sex, target, rows in REPLICA_QUOTA:
        call: list[object] = [planted(age=age, sex=sex, target=target, cp=2)] * rows
        calls.append(call)
    if arm == "bko":
        calls[0] = [*calls[0], PLANTED["code:cp"]]
    if arm == "bko-population":
        calls[0] = [*calls[0], BAD, BAD, BAD]
    if arm == "bko-frequency":
        calls[0] = [*calls[0], calls[0][0]]
        calls[6] = [planted(age=63, sex=1, target=0), *calls[6][1:]]
    return tuple(reply(call) for call in calls)


def replica_script() -> ScriptedSend:
    """Both holdouts' calls in harness order: each holdout's gap fills, then its replica fills."""
    arms = REPLICAS.arms[1:]
    return ScriptedSend(
        *(
            r
            for _ in REPLICAS.holdout_seeds
            for r in (
                *(g for arm in arms for g in holdout_replies(arm)),
                *(p for arm in REPLICAS.replica_arms for p in replica_replies(arm)),
            )
        )
    )


def test_every_bko_arm_fills_a_replica_counted_apart_and_replayed(tmp_path, caplog):
    send = replica_script()
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=REPLICAS)
    log = tmp_path / "calls.jsonl"
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    assert main(_score(log, tmp_path), context=ctx, out=out) == 0
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
    argv = _score(replayed, tmp_path, "--replay", str(log))
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
