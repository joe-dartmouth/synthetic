"""`bko.generate.reasoning`: a local thinking reply's reasoning length."""

from __future__ import annotations

import pytest

from bko.generate.local import reply_of
from bko.generate.reasoning import reasoning_chars, reasoning_text
from tests.test_generate_local import RECORD, completion


@pytest.mark.parametrize("thoughts", [7, ["a"], {"a": 1}, True])
def test_reasoning_that_is_not_text_counts_zero(thoughts):
    reply = reply_of(completion([RECORD], reasoning_content=thoughts))
    assert (reasoning_text(reply), reasoning_chars(reply)) == (None, 0)


def test_reasoning_text_counts_its_characters():
    reply = reply_of(completion([RECORD], reasoning_content="t" * 50))
    assert (reasoning_text(reply), reasoning_chars(reply)) == ("t" * 50, 50)
