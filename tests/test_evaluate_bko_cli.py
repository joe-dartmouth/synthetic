"""`python -m bko.evaluate` with a BKO arm: one call log, validity judged, API failures exit 2.

Every call goes to a `ScriptedSend`; no test reaches the API.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from typing import Final

import pandas as pd

from bko.evaluate.__main__ import Context, main
from bko.generate.errors import GenerationError
from tests._datasets_fixtures import pin
from tests._evaluate_fixtures import FAST_ARMS, SMALL
from tests._fusion_fixtures import young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, ScriptedSend, reply, uci
from tests._ontology_fixtures import RoutingOpener


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
