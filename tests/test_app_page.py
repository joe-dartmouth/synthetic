"""The page through AppTest over the real entry script: no test reaches the API or the network."""

from __future__ import annotations

import dataclasses
import threading
from typing import BinaryIO

import pytest
from streamlit.testing.v1 import AppTest

from bko.app.page import ACTION_LOCK, ANALYSIS_KEY, BUSY, CONTEXT_KEY, OUTCOME_KEY
from bko.app.steps import analyze, fill_and_score
from bko.datasets.errors import UnavailableError
from bko.gaps.definitions import Definition
from bko.generate.errors import GenerationError
from tests._app_fixtures import (
    analyzed,
    app_context,
    frame_with,
    launch,
    never,
    one_fill,
    settled,
)
from tests._generate_fixtures import ScriptedSend


def refusing(url: str, timeout: float) -> BinaryIO:
    """An opener that reaches nothing."""
    msg = f"{url} is unreachable"
    raise UnavailableError(msg)


def test_the_page_uses_the_injected_context_and_reaches_nothing_before_a_click(tmp_path):
    context, opener = app_context(tmp_path, send=never)
    at = launch(context)
    assert at.session_state[CONTEXT_KEY] is context
    assert opener.calls == []
    assert [box.label for box in at.selectbox] == ["Dataset"]


def test_settled_refuses_an_app_that_raised():
    at = AppTest.from_string("raise ValueError('boom')").run()
    with pytest.raises(AssertionError, match="boom"):
        settled(at)


def test_an_api_call_outside_a_live_test_fails_the_test(monkeypatch):
    """The paid guard's refusal escapes Streamlit's handler and kills the script thread.

    AppTest then raises rather than returning a run, so such a test never passes; the guard's
    message is on the thread's exception, not on what `run` raises.
    """
    escaped: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))
    script = (
        "import anthropic\n"
        "anthropic.Anthropic(api_key='unused').messages.create("
        "model='claude-haiku-5-5', max_tokens=1, messages=[{'role': 'user', 'content': 'x'}])\n"
    )
    with pytest.raises((KeyError, RuntimeError)):
        AppTest.from_string(script, default_timeout=5).run()
    assert [str(exc).split(":")[0] for exc in escaped] == [
        "a request outside a `live` test's body reached the Anthropic API"
    ]


def test_analyze_shows_the_flagged_cells_and_each_definitions_gaps(tmp_path):
    at = analyzed(app_context(tmp_path, send=never)[0])
    cells = frame_with(at, {"count", "status"})
    assert set(cells.loc[cells["status"] == "underrepresented", "age_band"]) == {"<55"}
    assert len(at.tabs) == len(Definition) + 1


def test_a_busy_app_refuses_a_second_action(tmp_path):
    at = launch(app_context(tmp_path, send=never)[0])
    ACTION_LOCK.acquire()
    try:
        at = settled(at.button(key="bko.analyze").click().run())
    finally:
        ACTION_LOCK.release()
    assert BUSY in [warning.value for warning in at.warning]
    assert ANALYSIS_KEY not in at.session_state


def test_a_failed_download_shows_its_message_not_a_traceback(tmp_path):
    at = analyzed(app_context(tmp_path, send=never, open_url=refusing)[0])
    assert at.error
    assert "is unreachable" in at.code[0].value


def _confirmed_fill(at: AppTest, runs: int = 0) -> AppTest:
    """`at` after choosing `bko`, ticking the confirmation for fill `runs`, and clicking Fill."""
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.checkbox(key=f"bko.confirm.bko.{runs}").check().run())
    return settled(at.button(key="bko.fill").click().run())


def test_a_paid_fill_needs_a_confirmation_and_makes_no_call_before_it(tmp_path):
    send = one_fill()
    at = analyzed(app_context(tmp_path, send=lambda: send)[0])
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    assert at.button(key="bko.fill").disabled is True
    assert send.calls == []
    assert not (tmp_path / "runs").exists()
    at = settled(at.checkbox(key="bko.confirm.bko.0").check().run())
    settled(at.button(key="bko.fill").click().run())
    assert len(send.calls) == 2


def test_a_rerun_after_a_paid_fill_pays_nothing_more(tmp_path):
    send = one_fill()
    at = _confirmed_fill(analyzed(app_context(tmp_path, send=lambda: send)[0]))
    at = settled(settled(at.run()).run())
    assert len(send.calls) == 2
    assert len(list((tmp_path / "runs").iterdir())) == 1
    assert "4. Results" in [header.value for header in at.header]


def test_a_second_click_needs_a_fresh_confirmation(tmp_path):
    """After a fill the box is a new, unticked one, so the button is disabled until it is ticked.

    AppTest refuses to click a disabled button, as a browser does; `if clicked and ready` covers a
    click that arrives anyway.
    """
    send = one_fill()
    at = _confirmed_fill(analyzed(app_context(tmp_path, send=lambda: send)[0]))
    assert at.button(key="bko.fill").disabled is True
    assert at.checkbox(key="bko.confirm.bko.1").value is False
    assert len(send.calls) == 2


def test_a_stopped_fill_shows_its_message_not_a_traceback(tmp_path):
    failure = GenerationError("no Anthropic credential: set ANTHROPIC_API_KEY", attempts=1)
    at = _confirmed_fill(analyzed(app_context(tmp_path, send=lambda: ScriptedSend(failure))[0]))
    assert at.error
    assert any("no Anthropic credential" in block.value for block in at.code)


def test_a_replayed_fill_pays_nothing(tmp_path):
    send = one_fill()
    context, _ = app_context(tmp_path, send=lambda: send)
    recorded = fill_and_score(analyze(context), "bko", context)
    at = analyzed(dataclasses.replace(context, send=never))
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.radio(key="bko.answers").set_value("replay").run())
    at = settled(at.selectbox(key="bko.log").select(recorded.folder.name).run())
    at = settled(at.button(key="bko.fill").click().run())
    assert at.session_state[OUTCOME_KEY].value.replayed is True


def test_replay_needs_a_log_for_the_chosen_arm(tmp_path):
    at = analyzed(app_context(tmp_path, send=never)[0])
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.radio(key="bko.answers").set_value("replay").run())
    assert "No call log to replay yet." in [info.value for info in at.info]
    assert at.button(key="bko.fill").disabled is True


def test_a_free_fill_ends_with_closure_scores_and_two_downloads(tmp_path):
    at = analyzed(app_context(tmp_path, send=never)[0])
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run())
    frame_with(at, {"closed", "still_open"})
    frame_with(at, {"definition", "status"})
    assert any({"none", "smote"} <= set(frame.value.columns) for frame in at.dataframe)
    assert len(at.get("download_button")) == 2


def test_changing_the_method_forgets_the_outcome(tmp_path):
    at = analyzed(app_context(tmp_path, send=never)[0])
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run())
    assert OUTCOME_KEY in at.session_state
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    assert OUTCOME_KEY not in at.session_state
