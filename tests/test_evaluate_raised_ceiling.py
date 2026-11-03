"""`score --resume --ceiling`: a spend stop resumed under a ceiling raised for that resume alone.

Every call goes to a `ScriptedSend`; no test reaches the API. Each run keeps one call in flight,
so the ceiling stops it at the call the spend reaches.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
from pathlib import Path

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.experiment import Experiment
from bko.generate.messages import SETTINGS, Reply, Usage
from bko.generate.spend import cost
from tests._evaluate_bko_fixtures import NARROW, context, score_argv
from tests._fusion_fixtures import young
from tests._generate_fixtures import ScriptedSend, reply


SPENT = cost(reply([]).usage, SETTINGS.model)
"""One scripted call's spend in USD."""


def _gap(cell: int, rows: int, times: int = 1) -> Reply:
    """One gap call's reply of `rows` records of `young(cell)`, billed `times` one call's tokens."""
    return reply([young(cell)] * rows, usage=Usage(1000 * times, 500 * times, 0, 0))


def _holdout(times: int = 1) -> tuple[Reply, Reply]:
    """One holdout's two gap calls, the first billed `times` one call's tokens."""
    return _gap(0, 12, times), _gap(1, 16)


def _lines(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _score(
    log: Path, tmp_path: Path, experiment: Experiment, send: ScriptedSend, *flags: str
) -> tuple[int, str]:
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=experiment)
    out = io.StringIO()
    argv = score_argv(log, tmp_path, "--in-flight", "1", *flags)
    return main(argv, context=ctx, out=out), out.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_ceiling_without_a_resume_is_refused_before_any_download(tmp_path, caplog):
    send = ScriptedSend()
    with caplog.at_level(logging.ERROR):
        code, out = _score(tmp_path / "c.jsonl", tmp_path, NARROW, send, "--ceiling", "5")
    assert (code, out) == (2, "")
    assert "--ceiling raises a resumed run's spend ceiling, so it needs --resume" in caplog.text
    assert send.calls == []


def test_a_raised_resume_stops_at_its_own_ceiling_and_resumes_again_raised(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _, text = _score(tmp_path / "paid.jsonl", tmp_path, NARROW, ScriptedSend(*_holdout() * 2))
    rules = dataclasses.replace(NARROW, projection_after=1, spend_ceiling=3 * SPENT)
    stopped = tmp_path / "stopped.jsonl"
    assert _score(stopped, tmp_path, rules, ScriptedSend(*_holdout()))[0] == 2
    assert (_lines(stopped)[-1]["stop"], _lines(stopped)[-1]["holdouts"]) == ("projection", 1)
    # 2 calls over 1 of 2 holdouts project to 4 calls; the raised 4.5 covers that, but the second
    # holdout's first call bills 3 calls' tokens, so the spend reaches 5 calls and stops there.
    first = tmp_path / "first.jsonl"
    raised = repr(4.5 * SPENT)
    send = ScriptedSend(*_holdout(times=3))
    code, out = _score(first, tmp_path, rules, send, "--resume", str(stopped), "--ceiling", raised)
    assert (code, out) == (2, "")
    assert len(send.calls) == 1
    end = _lines(first)[-1]
    assert (end["status"], end["stop"], end["holdouts"]) == ("failed", "ceiling", 1)
    assert end["usd"] == pytest.approx(5 * SPENT)
    assert end["live_usd"] == pytest.approx(3 * SPENT)
    record = {"sha256": _sha256(stopped), "calls": 2, "stop": "projection", "ceiling": 4.5 * SPENT}
    assert end["resume"] == record
    assert any(m.startswith("stop rule=ceiling usd=") for m in caplog.messages)
    # 5 calls over 1 holdout project to 10; a resume of the resume raised past that finishes.
    again = tmp_path / "again.jsonl"
    send = ScriptedSend(_gap(1, 16))
    argv = ("--resume", str(first), "--ceiling", repr(10.5 * SPENT))
    code, out = _score(again, tmp_path, rules, send, *argv)
    assert (code, out) == (0, text)
    send.assert_consumed()
    lines = _lines(again)
    record = {"sha256": _sha256(first), "calls": 3, "stop": "ceiling", "ceiling": 10.5 * SPENT}
    assert (lines[0]["resume"], lines[-1]["resume"]) == (record, record)
    assert lines[-1]["usd"] == pytest.approx(6 * SPENT)
    assert lines[-1]["live_usd"] == pytest.approx(SPENT)


def test_a_carried_spend_the_guard_scaled_over_more_holdouts_restops_on_the_projection(
    tmp_path, caplog
):
    """The guard scales the stopped spend over the holdouts it finished; the projection rule
    scales the resumed spend after `projection_after`, carried spend included. When spend a
    concurrent log dropped is carried, the second can exceed a ceiling the first allowed: the
    resume then stops loudly on the projection while it still replays, paying nothing."""
    caplog.set_level(logging.INFO)
    four = dataclasses.replace(
        NARROW, holdout_seeds=(0, 1, 2, 3), projection_after=1, spend_ceiling=8.5 * SPENT
    )
    stopped = tmp_path / "stopped.jsonl"
    send = ScriptedSend(*_holdout(), *_holdout(), *_holdout(times=5))
    assert _score(stopped, tmp_path, four, send)[0] == 2
    lines = _lines(stopped)
    assert (lines[-1]["stop"], lines[-1]["holdouts"]) == ("ceiling", 2)
    assert lines[-1]["usd"] == pytest.approx(9 * SPENT)
    # As if a concurrent segment before it had dropped calls worth 6 calls' spend.
    lines[0]["carried_usd"] = 6 * SPENT
    lines[-1]["usd"] += 6 * SPENT
    stopped.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    # The guard: 15 calls over 2 of 4 holdouts project to 30; 30.5 covers that. The projection
    # rule: 6 carried and 2 replayed over 1 holdout project to 32, past 30.5.
    resumed = tmp_path / "resumed.jsonl"
    live = ScriptedSend()
    argv = ("--resume", str(stopped), "--ceiling", repr(30.5 * SPENT))
    assert _score(resumed, tmp_path, four, live, *argv) == (2, "")
    assert live.calls == []
    end = _lines(resumed)[-1]
    assert (end["status"], end["stop"], end["live_usd"]) == ("failed", "projection", 0.0)
    assert any(m.startswith("stop rule=projection projected_usd=") for m in caplog.messages)
