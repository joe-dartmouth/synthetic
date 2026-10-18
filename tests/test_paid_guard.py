"""The paid-API guard sits on the SDK's one request sink, so an unmarked test cannot spend money.

`tests/conftest.py`'s collection hook skips a `live` test unless `--live` was passed, and that keys
on the marker a human remembered to apply. The guard beside it covers the class the marker cannot:
a test nobody marked that still reaches the Anthropic API. It patches `SyncAPIClient.request` and
`AsyncAPIClient.request`, which every resource method reaches, rather than a list of methods, so
these cases drive several methods and require the one refusal from each.

**Safe by construction, which matters because a broken guard would otherwise spend money here.**
Every client is built against a `base_url` on the loopback discard port with retries off, so a
guard that failed to fire meets a refused connection on this machine rather than a billable
request. That is also what makes the refusal assertion discriminating: without the guard these
cases fail with `APIConnectionError`, not with the guard's message.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

import anthropic
import pytest
from anthropic._base_client import AsyncAPIClient, SyncAPIClient
from anthropic.types import MessageParam


_REFUSAL = "reached the Anthropic API"
"""The part of the guard's refusal every case here requires."""

_NOWHERE = "http://127.0.0.1:9"
"""The loopback discard port: a request that escapes the guard is refused locally."""

_MODEL = "claude-opus-5-5"
"""Any model name; no request reaches a server to validate it."""

_MESSAGE: list[MessageParam] = [{"role": "user", "content": "hi"}]

_NOT_A_CREDENTIAL = "not-a-credential"
"""What every client here is built with; nothing ever presents it to a server."""


def _anthropic() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=_NOT_A_CREDENTIAL, base_url=_NOWHERE, max_retries=0)


def _create(client: anthropic.Anthropic) -> object:
    return client.messages.create(model=_MODEL, max_tokens=1, messages=_MESSAGE)


def _stream(client: anthropic.Anthropic) -> object:
    with client.messages.stream(model=_MODEL, max_tokens=1, messages=_MESSAGE) as stream:
        return stream.get_final_message()


def _count(client: anthropic.Anthropic) -> object:
    return client.messages.count_tokens(model=_MODEL, messages=_MESSAGE)


def _retrieve(client: anthropic.Anthropic) -> object:
    return client.models.retrieve(_MODEL)


_CALLS: dict[str, Callable[[anthropic.Anthropic], object]] = {
    "messages.create": _create,
    "messages.stream": _stream,
    "messages.count_tokens": _count,
    "models.retrieve": _retrieve,
}
"""Four resource methods. What they share is the sink, which is the claim: a method added to the
SDK later reaches the same `request` and is refused without this dict learning its name."""


@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
def test_every_method_of_the_direct_client_reaches_the_one_sink(
    call: Callable[[anthropic.Anthropic], object],
) -> None:
    """`messages.create` and its siblings each end in the refusal, not in a request."""
    with pytest.raises(pytest.fail.Exception, match=_REFUSAL):
        call(_anthropic())


def test_the_async_client_is_refused_too() -> None:
    """Nothing uses the async client today; the guard covers it so the first caller is safe."""
    client = anthropic.AsyncAnthropic(api_key=_NOT_A_CREDENTIAL, base_url=_NOWHERE, max_retries=0)

    async def create() -> object:
        return await client.messages.create(model=_MODEL, max_tokens=1, messages=_MESSAGE)

    with pytest.raises(pytest.fail.Exception, match=_REFUSAL):
        asyncio.run(create())


def test_no_client_class_overrides_the_patched_sink() -> None:
    """The structural half: a client class defining its own `request` would walk past the patch.

    The cases above catch that for the four methods they drive. This catches it for every method
    and every exported client class at once, and names the class that moved if an SDK release
    overrides the sink. `Anthropic` and `AsyncAnthropic` are the control that discovery found
    something; the names of the two base methods show the guard, not the SDK, is installed there.
    """
    found = {
        value
        for value in vars(anthropic).values()
        if isinstance(value, type) and issubclass(value, (SyncAPIClient, AsyncAPIClient))
    }
    assert {anthropic.Anthropic, anthropic.AsyncAnthropic} <= found, found
    for cls in found:
        base = SyncAPIClient if issubclass(cls, SyncAPIClient) else AsyncAPIClient
        assert cls.request is base.request, (
            f"{cls.__name__} resolves `request` somewhere other than {base.__name__}, so patching "
            "the base no longer guards it; patch the class that now defines it"
        )
    assert SyncAPIClient.request.__name__ == "guarded"
    assert AsyncAPIClient.request.__name__ == "guarded_async"


def test_undoing_the_tests_own_monkeypatch_does_not_disarm_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard patches through a patcher of its own, so a test's `monkeypatch.undo()` keeps it."""
    monkeypatch.undo()

    with pytest.raises(pytest.fail.Exception, match=_REFUSAL):
        _create(_anthropic())


@pytest.fixture(scope="module")
def module_scoped_attempt() -> BaseException | None:
    """What a module-scoped fixture meets when it reaches the API during setup.

    Returns:
        The exception the request raised, or `None` if it returned.
    """
    try:
        _create(_anthropic())
    except BaseException as exc:
        return exc
    return None


def test_a_module_scoped_fixture_is_refused_too(
    module_scoped_attempt: BaseException | None,
) -> None:
    """Wide-scope fixtures run before any function-scoped patch, so the guard is session-wide.

    Requests are let through only inside a `live` test's body; a module-scoped fixture's setup is
    outside it.
    """
    assert isinstance(module_scoped_attempt, pytest.fail.Exception), (
        f"a module-scoped fixture's request met {type(module_scoped_attempt).__name__}, not the "
        "guard, so setup code outside a test body can spend money"
    )
    assert _REFUSAL in str(module_scoped_attempt)


def test_without_live_the_base_url_is_the_discard_port(request: pytest.FixtureRequest) -> None:
    """Without `--live` a client built with no `base_url` points at the discard port.

    A patch does not cross a process boundary, so this is what keeps a subprocess (a CLI test, say)
    from spending: it inherits the variable, and the SDK reads it before any profile.
    """
    if request.config.getoption("--live"):
        pytest.skip("under `--live` the base URL is left alone")
    assert os.environ["ANTHROPIC_BASE_URL"] == _NOWHERE
    assert str(anthropic.Anthropic(api_key="x").base_url).startswith(_NOWHERE)
