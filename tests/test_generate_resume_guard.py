"""The resume guard's refusals, its raised and inherited ceilings, and `log_ends`' edge cases."""

from __future__ import annotations

import contextlib
import json
import logging
import math
from pathlib import Path

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.generate.calls import ask
from bko.generate.errors import ReplayError
from bko.generate.log import Answered, CallLog
from bko.generate.messages import SETTINGS
from bko.generate.replay import Resume
from bko.generate.resume_guard import (
    TAIL,
    Ceiling,
    Raised,
    log_ends,
    ran_under,
    refuse_unresumable,
)
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply


PATH = Path("stopped.jsonl")
RUN = {"kind": "run", "schema": 2, "run": "e6-thina"}
END = {"kind": "end", "stop": "ceiling", "usd": 2.0, "live_usd": 1.5, "holdouts": 4}
RAISED = Ceiling(10.0, 10)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("usd", None),
        ("usd", True),
        ("usd", "2.0"),
        ("usd", -0.01),
        ("usd", math.inf),
        ("usd", math.nan),
        ("live_usd", None),
        ("live_usd", False),
        ("live_usd", -1),
    ],
)
def test_a_spend_stop_with_no_spend_of_at_least_zero_is_refused(key, value):
    with pytest.raises(ReplayError, match="records no usd and live_usd spend of at least 0"):
        refuse_unresumable(PATH, RUN, {**END, key: value}, RAISED)


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "4"])
def test_a_spend_stop_with_no_whole_count_of_holdouts_is_refused(value):
    with pytest.raises(ReplayError, match="records no finished holdouts"):
        refuse_unresumable(PATH, RUN, {**END, "holdouts": value}, RAISED)


def test_a_spend_stop_without_its_holdouts_key_is_refused():
    end = {key: value for key, value in END.items() if key != "holdouts"}
    with pytest.raises(ReplayError, match="records no finished holdouts: None"):
        refuse_unresumable(PATH, RUN, end, RAISED)


def test_a_covered_spend_stop_returns_what_it_read():
    found = refuse_unresumable(PATH, RUN, END, RAISED, name="e6-thina")
    assert found == Raised("ceiling", 2.0, 1.5, 4, 5.0, 10.0)


def test_a_log_of_another_named_run_is_refused_whatever_its_stop():
    for end in (END, None, {"kind": "end", "stop": "idle"}):
        with pytest.raises(ReplayError, match="holds the calls of the run 'e6-thina', not 'e6"):
            refuse_unresumable(PATH, RUN, end, RAISED, name="e6-thinb")


def test_a_spend_stop_whose_log_names_no_run_is_refused_under_a_name():
    with pytest.raises(ReplayError, match="names no run, so it cannot be checked"):
        refuse_unresumable(PATH, {"kind": "run"}, END, RAISED, name="e6-thina")


def test_another_stop_whose_log_names_no_run_still_resumes():
    assert refuse_unresumable(PATH, {"kind": "run"}, None, name="e6-thina") is None


@pytest.mark.parametrize("ceiling", [{"ceiling": 0}, {"ceiling": True}, {"ceiling": "9"}])
def test_a_malformed_recorded_ceiling_is_refused(ceiling):
    with pytest.raises(ReplayError, match="records the raised ceiling"):
        ran_under(PATH, {"resume": {"sha256": "ab", "calls": 1, **ceiling}})


def test_a_log_with_no_recorded_ceiling_runs_under_the_named_runs():
    assert ran_under(PATH, {"resume": None}) is None
    assert ran_under(PATH, {"resume": {"sha256": "ab", "calls": 1}}) is None
    recorded = ran_under(PATH, {"resume": {"sha256": "ab", "calls": 1, "ceiling": 12.5}})
    assert recorded == pytest.approx(12.5)


def _resumed(tmp_path: Path, answered: Answered, end: dict | None = None) -> Path:
    """A resumed log of one call, answered from `answered`, its end line updated with `end`."""
    path = tmp_path / "resumed.jsonl"
    with (
        contextlib.suppress(KeyboardInterrupt),
        CallLog(path, SETTINGS, answered=answered, run="e6-thina", frame="uci") as log,
    ):
        ask(GUIDANCE, 2, send=ScriptedSend(reply([young(1)] * 2)), log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[-1].update(end or {})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def test_a_resume_of_a_resume_takes_a_spend_stop_under_a_ceiling_raised_again(tmp_path):
    earlier = Answered("resume", "ab" * 32, 1, 0.5, "projection", 6.0)
    path = _resumed(tmp_path, earlier, {"status": "failed", **END})
    resume = Resume(path, ScriptedSend(), ceiling=RAISED, run="e6-thina")
    assert resume.raised == Raised("ceiling", 2.0, 1.5, 4, 5.0, 10.0)
    assert resume.ceiling == pytest.approx(10.0)
    assert resume.answered == Answered("resume", resume.digest, 1, 0.5, "ceiling", 10.0)


def test_a_resume_of_a_raised_resume_that_stopped_otherwise_inherits_its_ceiling(tmp_path):
    earlier = Answered("resume", "ab" * 32, 1, 0.5, "ceiling", 6.0)
    path = _resumed(tmp_path, earlier, {"status": "failed", "stop": "idle", "retryable": True})
    resume = Resume(path, ScriptedSend(), run="e6-thina")
    assert (resume.raised, resume.ceiling) == (None, 6.0)
    assert resume.answered == Answered("resume", resume.digest, 1, 0.5, None, 6.0)
    record = tmp_path / "again.jsonl"
    with CallLog(record, SETTINGS, answered=resume.answered, run="e6-thina", frame="uci"):
        pass
    run_line = json.loads(record.read_text(encoding="utf-8").splitlines()[0])
    assert run_line["resume"] == {"sha256": resume.digest, "calls": 1, "ceiling": 6.0}


def _log(tmp_path: Path, *lines: dict, cut: bytes = b"") -> Path:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines) + cut)
    return path


def test_log_ends_refuses_a_log_that_is_not_schema_2(tmp_path):
    with pytest.raises(ReplayError, match="is not a schema 2 call log"):
        log_ends(_log(tmp_path, {**RUN, "schema": 1}))


def test_log_ends_refuses_an_empty_log(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b"")
    with pytest.raises(ReplayError, match="is not a schema 2 call log"):
        log_ends(path)


def test_log_ends_reads_a_log_shorter_than_its_tail(tmp_path):
    path = _log(tmp_path, RUN, END)
    assert path.stat().st_size < TAIL
    assert log_ends(path) == (RUN, END)


def test_log_ends_reads_the_end_line_of_a_log_its_tail_cuts_into(tmp_path):
    filler = {"kind": "reply", "call": 1, "response": "x" * (2 * TAIL)}
    path = _log(tmp_path, RUN, filler, END)
    assert path.stat().st_size > 2 * TAIL
    assert log_ends(path) == (RUN, END)


def test_log_ends_finds_no_end_line_that_is_not_last(tmp_path):
    assert log_ends(_log(tmp_path, RUN, END, {"kind": "request", "call": 2})) == (RUN, None)


def test_log_ends_finds_no_end_line_in_a_log_of_its_run_line_alone(tmp_path):
    assert log_ends(_log(tmp_path, RUN)) == (RUN, None)


def test_log_ends_warns_of_a_cut_last_line_and_finds_no_end_line(tmp_path, caplog):
    path = _log(tmp_path, RUN, END, cut=b'{"kind": "rep')
    with caplog.at_level(logging.WARNING):
        assert log_ends(path) == (RUN, None)
    assert f"call log {path} ends in a cut line of 13 bytes, not an end line" in caplog.text
