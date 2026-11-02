"""`bko.generate.replay.Replay`: a call log's calls answered in order, each checked."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Final

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.errors import StopRuleError
from bko.evaluate.waves import spendable
from bko.generate.calls import ask
from bko.generate.errors import GenerationError, ReplayError
from bko.generate.guidance import Guidance
from bko.generate.log import Answered, CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.prompt import render
from bko.generate.replay import Replay, Resume
from bko.generate.resume_guard import Ceiling, Raised
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply


def call_for(guidance: Guidance, rows: int) -> Call:
    """The call `ask` sends for `rows` records of `guidance`'s cell."""
    return Call(
        SETTINGS,
        guidance.sex,
        guidance.label,
        *render(guidance, rows, SETTINGS.guidance, frame=UCI_FRAME),
        frame=UCI_FRAME,
    )


def logged(tmp_path: Path, *script: Reply | GenerationError) -> Path:
    """A call log of one `ask` for 2 records of `GUIDANCE` per scripted outcome."""
    path = tmp_path / "paid.jsonl"
    with CallLog(path, SETTINGS, frame="uci") as log:
        for outcome in script:
            with contextlib.suppress(GenerationError):
                ask(GUIDANCE, 2, send=ScriptedSend(outcome), log=log, frame=UCI_FRAME)
    return path


def test_a_replayed_reply_equals_the_logged_reply(tmp_path):
    original = dataclasses.replace(reply([young(1), young(1)]), attempts=2)
    replay = Replay(logged(tmp_path, original))
    with CallLog(tmp_path / "again.jsonl", SETTINGS, answered=replay.answered, frame="uci") as log:
        batch = ask(GUIDANCE, 2, send=replay, log=log, frame=UCI_FRAME)
    assert batch.reply == original
    assert replay.remaining == 0


def test_replay_answers_one_request_in_the_logged_order(tmp_path):
    first = reply([young(1), young(1)], request_id="req_1")
    second = reply([young(1), young(1)], request_id="req_2")
    replay = Replay(logged(tmp_path, first, second))
    call = call_for(GUIDANCE, 2)
    assert [replay(call).request_id, replay(call).request_id] == ["req_1", "req_2"]


def test_a_logged_failure_is_raised_again(tmp_path):
    failure = GenerationError("the API could not be reached: X", attempts=3)
    replay = Replay(logged(tmp_path, failure))
    with pytest.raises(GenerationError, match="could not be reached") as raised:
        replay(call_for(GUIDANCE, 2))
    assert raised.value.attempts == 3


def test_a_call_that_differs_is_refused(tmp_path):
    replay = Replay(logged(tmp_path, reply([young(1), young(1)])))
    with pytest.raises(ReplayError, match="differs"):
        replay(call_for(GUIDANCE, 3))


def test_a_call_past_the_end_is_refused(tmp_path):
    replay = Replay(logged(tmp_path))
    with pytest.raises(ReplayError, match="no call left"):
        replay(call_for(GUIDANCE, 2))


def test_a_log_of_another_schema_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    lines = path.read_text().splitlines()
    run = json.loads(lines[0]) | {"schema": 1}
    path.write_text("\n".join([json.dumps(run), *lines[1:]]) + "\n")
    with pytest.raises(ReplayError, match="schema 2"):
        Replay(path)


def test_a_line_that_is_not_json_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path.write_text(path.read_text() + "{\n")
    with pytest.raises(ReplayError, match="not JSON"):
        Replay(path)


def test_a_request_with_no_answer_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    lines = path.read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["kind"] != "reply"]
    path.write_text("\n".join(kept) + "\n")
    with pytest.raises(ReplayError, match="no reply or error"):
        Replay(path)


def rewritten(path: Path, edit) -> Path:
    """Rewrite `path`'s JSON lines after `edit` changed the list in place."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(lines)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def first(lines: list[dict], kind: str) -> dict:
    """The first line of `kind`."""
    return next(line for line in lines if line["kind"] == kind)


def test_a_line_that_is_not_an_object_is_refused(tmp_path):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path.write_text(path.read_text() + "[1]\n")
    with pytest.raises(ReplayError, match="not a JSON object"):
        Replay(path)


def test_an_empty_log_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ReplayError, match="not a schema 2 call log"):
        Replay(path)


def test_a_log_whose_first_line_is_not_its_run_line_is_refused(tmp_path):
    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), lambda lines: lines.reverse())
    with pytest.raises(ReplayError, match="not a schema 2 call log"):
        Replay(path)


@pytest.mark.parametrize("kind", ["request", "reply"])
@pytest.mark.parametrize("call", [None, "1", 1.0, True])
def test_a_line_without_a_whole_number_call_is_refused(tmp_path, kind, call):
    def edit(lines: list[dict]) -> None:
        line = first(lines, kind)
        if call is None:
            del line["call"]
        else:
            line["call"] = call

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="no whole-number call"):
        Replay(path)


@pytest.mark.parametrize("kind", ["request", "reply"])
def test_a_repeated_call_is_refused_not_collapsed(tmp_path, kind):
    def edit(lines: list[dict]) -> None:
        lines.insert(-1, dict(first(lines, kind)))

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="more than one"):
        Replay(path)


def test_calls_that_skip_an_index_are_refused(tmp_path):
    def edit(lines: list[dict]) -> None:
        for line in lines[1:-1]:
            line["call"] += 1

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="not calls 1 to 1"):
        Replay(path)


def test_an_answer_with_no_request_is_refused(tmp_path):
    def edit(lines: list[dict]) -> None:
        lines.insert(-1, first(lines, "reply") | {"call": 2})

    path = rewritten(logged(tmp_path, reply([young(1), young(1)])), edit)
    with pytest.raises(ReplayError, match="no request line"):
        Replay(path)


@pytest.mark.parametrize(
    ("kind", "key"),
    [
        ("request", "params"),
        ("reply", "usage"),
        ("reply", "response"),
        ("reply", "attempts"),
        ("reply", "request_id"),
        ("error", "error"),
        ("error", "attempts"),
    ],
)
def test_a_line_missing_a_field_is_refused(tmp_path, kind, key):
    outcome = GenerationError("x") if kind == "error" else reply([young(1), young(1)])
    path = rewritten(logged(tmp_path, outcome), lambda lines: first(lines, kind).pop(key))
    with pytest.raises(ReplayError, match=f"call 1 of {path}"):
        Replay(path)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0},
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "server_tool_use": 0,
        },
        {
            "input_tokens": 1.5,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    ],
)
def test_a_reply_whose_usage_is_not_usage_is_refused(tmp_path, usage):
    path = logged(tmp_path, reply([young(1), young(1)]))
    path = rewritten(path, lambda lines: first(lines, "reply").update(usage=usage))
    with pytest.raises(ReplayError, match="usage"):
        Replay(path)


def stopped(tmp_path: Path, *script: Reply | GenerationError) -> Path:
    """A log of `logged`'s calls whose run was then interrupted, so its end is not complete."""
    path = tmp_path / "stopped.jsonl"
    with contextlib.suppress(KeyboardInterrupt), CallLog(path, SETTINGS, frame="uci") as log:
        for outcome in script:
            with contextlib.suppress(GenerationError):
                ask(GUIDANCE, 2, send=ScriptedSend(outcome), log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    return path


def answer() -> Reply:
    return reply([young(1), young(1)])


LOST: Final = GenerationError("the API could not be reached: X")


def test_resume_answers_the_answered_calls_then_sends_live(tmp_path):
    live = ScriptedSend(answer())
    resume = Resume(stopped(tmp_path, answer(), LOST), live)
    assert resume.answered == Answered("resume", resume.digest, 1)
    call = call_for(GUIDANCE, 2)
    resume(call)
    assert live.calls == []
    resume(call)
    live.assert_consumed()


def test_resume_drops_a_cut_final_line_and_an_unanswered_call(tmp_path):
    path = stopped(tmp_path, answer())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "request", "call": 2, "params": {}}) + "\n")
        handle.write('{"kind": "rep')
    assert Resume(path, ScriptedSend()).answered.calls == 1


def test_resume_drops_a_last_call_that_has_a_reply_and_an_error(tmp_path):
    path = stopped(tmp_path, answer(), answer())
    interrupt = {"kind": "error", "call": 2, "error": "KeyboardInterrupt", "attempts": None}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**interrupt, "at": "x"}) + "\n")
    assert Resume(path, ScriptedSend()).answered.calls == 1


def test_resume_refuses_an_error_before_the_last_call(tmp_path):
    with pytest.raises(ReplayError, match="failed before the run's last call"):
        Resume(stopped(tmp_path, LOST, answer()), ScriptedSend())


def test_resume_refuses_a_replays_log(tmp_path):
    source = Replay(logged(tmp_path, answer()))
    path = tmp_path / "replaying.jsonl"
    with (
        contextlib.suppress(KeyboardInterrupt),
        CallLog(path, SETTINGS, answered=source.answered, frame="uci") as log,
    ):
        ask(GUIDANCE, 2, send=source, log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    with pytest.raises(ReplayError, match=f"replays the log with sha256 {source.digest}"):
        Resume(path, ScriptedSend())


def test_resume_refuses_a_rehearsals_log_and_replay_reads_it(tmp_path):
    path = tmp_path / "rehearsal.jsonl"
    with CallLog(path, SETTINGS, rehearsal=True, frame="uci") as log:
        ask(GUIDANCE, 2, send=ScriptedSend(answer()), log=log, frame=UCI_FRAME)
    with pytest.raises(ReplayError, match="rehearsal's log"):
        Resume(path, ScriptedSend())
    assert Replay(path).remaining == 1


def _stopped_by(tmp_path: Path, rule: str | None) -> Path:
    """A one-call log whose end line names `rule` under `stop`, or has no `stop` when None."""
    path = logged(tmp_path, answer())

    def edit(lines: list[dict]) -> None:
        lines[-1]["status"] = "failed"
        if rule is None:
            del lines[-1]["stop"]
        else:
            lines[-1]["stop"] = rule

    return rewritten(path, edit)


def _spent(
    path: Path, *, usd: float = 2.0, live_usd: float = 1.5, holdouts: int | None = 4
) -> Path:
    """`path` with its end line's spend and its finished holdouts set, or none when None."""

    def edit(lines: list[dict]) -> None:
        lines[-1].update(usd=usd, live_usd=live_usd)
        if holdouts is not None:
            lines[-1]["holdouts"] = holdouts

    return rewritten(path, edit)


@pytest.mark.parametrize("rule", ["projection", "ceiling"])
def test_resume_refuses_a_spend_stop_with_no_ceiling_to_read(tmp_path, rule):
    with pytest.raises(ReplayError, match="resumes only under the named run's ceiling"):
        Resume(_spent(_stopped_by(tmp_path, rule)), ScriptedSend())


@pytest.mark.parametrize("rule", ["projection", "ceiling"])
def test_resume_takes_a_spend_stop_whose_raised_ceiling_covers_its_projection(tmp_path, rule):
    # 2.0 USD over 4 of 10 holdouts projects to 5.0 USD; a ceiling above that resumes.
    resume = Resume(_spent(_stopped_by(tmp_path, rule)), ScriptedSend(), ceiling=Ceiling(5.01, 10))
    assert resume.raised == Raised(rule, 2.0, 1.5, 4, 5.0, 5.01)
    assert (resume.answered.stop, resume.answered.ceiling) == (rule, 5.01)
    assert resume.remaining == 1


@pytest.mark.parametrize("rule", ["projection", "ceiling"])
def test_resume_refuses_a_spend_stop_whose_ceiling_does_not_cover_its_projection(tmp_path, rule):
    path = _spent(_stopped_by(tmp_path, rule))
    message = (
        f"{path} was stopped by the {rule} rule at a logged spend of 2.0000 USD "
        "(live 1.5000 USD) over 4 holdouts, which projects to 5.0000 USD over 10; the named "
        "run's ceiling of 5.00 USD does not cover it, so raise the ceiling past the projection"
    )
    with pytest.raises(ReplayError) as raised:
        Resume(path, ScriptedSend(), ceiling=Ceiling(5.0, 10))
    assert str(raised.value) == message


def test_a_ceiling_stop_inside_its_first_holdout_projects_from_one(tmp_path):
    path = _spent(_stopped_by(tmp_path, "ceiling"), holdouts=0)
    with pytest.raises(ReplayError, match=r"projects to 20\.0000 USD over 10"):
        Resume(path, ScriptedSend(), ceiling=Ceiling(19.0, 10))
    raised = Resume(path, ScriptedSend(), ceiling=Ceiling(21.0, 10)).raised
    assert raised is not None
    assert raised.projected == pytest.approx(20.0)


def test_a_spend_stop_never_projects_below_its_logged_spend(tmp_path):
    path = _spent(_stopped_by(tmp_path, "ceiling"), holdouts=12)
    with pytest.raises(ReplayError, match=r"projects to 2\.0000 USD over 10"):
        Resume(path, ScriptedSend(), ceiling=Ceiling(2.0, 10))


@pytest.mark.parametrize("rule", ["projection", "ceiling"])
def test_resume_refuses_a_spend_stop_that_records_no_holdouts(tmp_path, rule):
    path = _spent(_stopped_by(tmp_path, rule), holdouts=None)
    with pytest.raises(ReplayError, match="records no finished holdouts"):
        Resume(path, ScriptedSend(), ceiling=Ceiling(100.0, 10))


def test_a_raised_ceiling_never_resumes_a_rehearsal_or_a_replay(tmp_path):
    path = tmp_path / "rehearsal.jsonl"
    with CallLog(path, SETTINGS, rehearsal=True, frame="uci") as log:
        ask(GUIDANCE, 2, send=ScriptedSend(answer()), log=log, frame=UCI_FRAME)

    def stop(lines: list[dict]) -> None:
        lines[-1].update(status="failed", stop="ceiling", holdouts=1)

    rewritten(path, stop)
    with pytest.raises(ReplayError, match="rehearsal's log"):
        Resume(path, ScriptedSend(), ceiling=Ceiling(1000.0, 10))
    replaying = rewritten(path, lambda lines: lines[0].update(rehearsal=False, replay="cd" * 32))
    with pytest.raises(ReplayError, match="replays the log"):
        Resume(replaying, ScriptedSend(), ceiling=Ceiling(1000.0, 10))


@pytest.mark.parametrize("rule", ["idle", None])
def test_another_stop_resumes_with_no_raised_record(tmp_path, rule):
    resume = Resume(_stopped_by(tmp_path, rule), ScriptedSend(), ceiling=Ceiling(1.0, 10))
    assert resume.raised is None
    assert resume.answered == Answered("resume", resume.digest, 1)


@pytest.mark.parametrize("rule", ["idle", None])
def test_resume_accepts_an_idle_stop_and_a_log_that_predates_the_stop_field(tmp_path, rule):
    assert Resume(_stopped_by(tmp_path, rule), ScriptedSend()).remaining == 1


def test_resume_refuses_a_resume_that_stopped_while_it_replayed(tmp_path):
    path = tmp_path / "resumed.jsonl"
    earlier = Answered("resume", "ab" * 32, 3)
    with (
        contextlib.suppress(KeyboardInterrupt),
        CallLog(path, SETTINGS, answered=earlier, frame="uci") as log,
    ):
        ask(GUIDANCE, 2, send=ScriptedSend(answer()), log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    with pytest.raises(ReplayError, match="ab" * 32):
        Resume(path, ScriptedSend())


def test_resuming_a_complete_log_answers_every_call_and_sends_nothing(tmp_path):
    live = ScriptedSend()
    resume = Resume(logged(tmp_path, answer()), live)
    resume(call_for(GUIDANCE, 2))
    assert resume.remaining == 0
    assert live.calls == []


def test_replay_still_refuses_a_cut_final_line(tmp_path):
    path = logged(tmp_path, answer())
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "rep')
    with pytest.raises(ReplayError, match="not JSON lines"):
        Replay(path)


def test_resume_accepts_a_resume_that_holds_every_call_it_replayed(tmp_path):
    path = tmp_path / "resumed.jsonl"
    earlier = Answered("resume", "ab" * 32, 1)
    with (
        contextlib.suppress(KeyboardInterrupt),
        CallLog(path, SETTINGS, answered=earlier, frame="uci") as log,
    ):
        ask(GUIDANCE, 2, send=ScriptedSend(answer()), log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    assert Resume(path, ScriptedSend()).answered.calls == 1


@pytest.mark.parametrize(
    "prior",
    [
        "x",
        {},
        {"sha256": "ab", "calls": "3"},
        {"sha256": 1, "calls": 1},
        {"sha256": "ab", "calls": True},
    ],
    ids=["text", "empty", "text calls", "number sha256", "bool calls"],
)
def test_resume_refuses_a_malformed_resume_field(tmp_path, prior):
    path = stopped(tmp_path, answer())
    lines = path.read_text(encoding="utf-8").splitlines()
    run = json.loads(lines[0]) | {"resume": prior}
    path.write_text("\n".join([json.dumps(run), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="resume"):
        Resume(path, ScriptedSend())


def test_resuming_a_log_with_no_calls_sends_every_call_live(tmp_path, caplog):
    live = ScriptedSend(answer())
    resume = Resume(stopped(tmp_path), live)
    assert resume.answered.calls == 0
    caplog.set_level(logging.INFO)
    resume(call_for(GUIDANCE, 2))
    live.assert_consumed()
    assert "resume live from call 1" in caplog.text


def test_resume_refuses_a_call_that_differs_while_it_replays(tmp_path):
    resume = Resume(stopped(tmp_path, answer(), answer()), ScriptedSend())
    with pytest.raises(ReplayError, match="differs"):
        resume(call_for(GUIDANCE, 3))


def test_resume_warns_of_what_it_drops(tmp_path, caplog):
    path = stopped(tmp_path, answer(), LOST)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "rep')
    with caplog.at_level(logging.WARNING):
        resume = Resume(path, ScriptedSend())
    assert resume.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert f"resume dropped a cut final line of 13 bytes from {path}" in caplog.text
    assert f"resume dropped call 2 of {path}, whose answer lines are ['error']" in caplog.text


def concurrent(tmp_path: Path, *script: Reply | GenerationError) -> Path:
    """`stopped`'s log of `script`, written by a run that kept 8 calls in flight."""
    path = tmp_path / "concurrent.jsonl"
    with (
        contextlib.suppress(KeyboardInterrupt),
        CallLog(path, SETTINGS, in_flight=8, frame="uci") as log,
    ):
        for outcome in script:
            with contextlib.suppress(GenerationError):
                ask(GUIDANCE, 2, send=ScriptedSend(outcome), log=log, frame=UCI_FRAME)
        raise KeyboardInterrupt
    return path


def _replied(path: Path, index: int) -> float:
    """The `usd` of call `index`'s reply line in the log at `path`."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return next(line["usd"] for line in lines if line["kind"] == "reply" and line["call"] == index)


def test_resume_drops_calls_after_the_first_unanswered_one(tmp_path, caplog):
    path = concurrent(tmp_path, *[answer()] * 41, LOST, answer())
    live = ScriptedSend(answer(), answer())
    with caplog.at_level(logging.WARNING):
        resume = Resume(path, live)
    assert resume.answered.calls == 41
    drops = [m for m in caplog.messages if m.startswith("resume dropped call")]
    assert drops == [
        f"resume dropped call 42 of {path}, whose answer lines are ['error']; it is sent again",
        f"resume dropped call 43 of {path}, whose answer lines are ['reply']; it is sent again",
    ]
    call = call_for(GUIDANCE, 2)
    for _ in range(41):
        resume(call)
    assert live.calls == []
    resume(call)
    resume(call)
    live.assert_consumed()


def test_resume_carries_the_spend_of_dropped_calls(tmp_path):
    path = concurrent(tmp_path, *[answer()] * 41, LOST, answer())
    resume = Resume(path, ScriptedSend())
    carried = _replied(path, 43)
    assert resume.answered.carried_usd == pytest.approx(carried)
    resumed = tmp_path / "resumed.jsonl"
    with CallLog(resumed, SETTINGS, answered=resume.answered, frame="uci") as log:
        assert log.usd == pytest.approx(carried)
        with pytest.raises(StopRuleError, match="reached its ceiling"):
            spendable(log, carried)
    run = json.loads(resumed.read_text(encoding="utf-8").splitlines()[0])
    assert run["carried_usd"] == pytest.approx(carried)


def test_a_resume_of_a_resume_keeps_the_spend_its_log_carried(tmp_path):
    path = concurrent(tmp_path, answer(), LOST, answer())
    rewritten(path, lambda lines: lines[0].update(carried_usd=0.5))
    resume = Resume(path, ScriptedSend())
    assert resume.answered.carried_usd == pytest.approx(0.5 + _replied(path, 3))


def test_a_resume_of_a_resume_at_in_flight_1_keeps_the_spend_its_log_carried(tmp_path):
    path = stopped(tmp_path, answer(), LOST)
    rewritten(path, lambda lines: lines[0].update(carried_usd=0.5))
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["in_flight"] == 1
    assert Resume(path, ScriptedSend()).answered.carried_usd == pytest.approx(0.5)


def test_an_old_log_with_an_error_before_its_last_call_is_still_refused(tmp_path):
    path = concurrent(tmp_path, LOST, answer())
    rewritten(path, lambda lines: lines[0].pop("in_flight"))
    with pytest.raises(ReplayError, match="failed before the run's last call"):
        Resume(path, ScriptedSend())
