"""The page through AppTest over the real entry script: no test reaches the API or the network."""

from __future__ import annotations

import threading

import pytest
from streamlit.testing.v1 import AppTest

from bko.app.page import CONTEXT_KEY
from tests._app_fixtures import app_context, launch, never, settled


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
