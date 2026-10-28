"""The SDK adapter: a message read into a `Reply`, and the SDK's failures into `GenerationError`."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import traceback
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import anthropic
import pytest

import bko.generate.api as api_module
from bko.generate.api import MAX_RETRIES, Attempts, async_send_with, reply_of, send_with
from bko.generate.errors import GenerationError
from bko.generate.messages import CALL_NUMBER, SETTINGS, Call, Send, Usage
from tests._generate_fixtures import message, records_text, request, response


CALL = Call(SETTINGS, 0, 1, "the system", "the user")
"""One call's request."""

SENTINEL = "sk-ant-SENTINEL"
"""A key an error body echoes."""

CREDENTIAL_ERROR = (
    '"Could not resolve authentication method. Expected one of api_key, auth_token, or '
    "credentials to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be "
    'explicitly omitted"'
)
"""The `TypeError` text SDK 1.13.0's `_validate_headers` raises at request time."""


def _client() -> anthropic.Anthropic:
    """A real client with a dummy key; the run's base URL is the guard's discard port."""
    return anthropic.Anthropic(api_key="not-a-credential", max_retries=0)


def _raising(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> anthropic.Anthropic:
    """A client whose `messages.create` raises `exc`."""
    client = _client()

    def create(**_: object) -> anthropic.types.Message:
        raise exc

    monkeypatch.setattr(client.messages, "create", create)
    return client


def test_reply_of_reads_request_id_model_stop_reason_text_usage_and_raw():
    built = message(records_text([]), stop_reason="max_tokens", input_tokens=7, output_tokens=9)
    extra = anthropic.types.TextBlock(type="text", text=" more")
    two = built.model_copy(update={"content": [*built.content, extra]})
    two._request_id = "req_two"
    reply = reply_of(two)
    assert reply.request_id == "req_two"
    assert reply.model == SETTINGS.model
    assert reply.stop_reason == "max_tokens"
    assert reply.text == records_text([]) + " more"
    assert reply.usage == Usage(7, 9, 0, 0)
    assert reply.raw == two.to_dict(mode="json")


def test_reply_of_reads_cache_writes_and_reads_each_into_its_own_field():
    built = message(records_text([]), cache_creation_input_tokens=3, cache_read_input_tokens=5)
    assert reply_of(built).usage == Usage(1000, 500, 3, 5)


def test_reply_of_gives_no_text_without_a_text_block():
    assert reply_of(message(None, stop_reason="refusal")).text is None


def test_send_with_passes_the_call_fields(monkeypatch):
    client = _client()
    seen: list[dict[str, object]] = []

    def create(**kwargs: object) -> anthropic.types.Message:
        seen.append(kwargs)
        return message(records_text([]))

    monkeypatch.setattr(client.messages, "create", create)
    reply = send_with(client)(CALL)
    assert seen == [CALL.params()]
    assert reply.request_id == "req_test"


def test_send_with_on_a_real_client_is_refused_by_the_guard():
    with pytest.raises(pytest.fail.Exception, match="reached the Anthropic API"):
        send_with(_client())(CALL)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (
            lambda: anthropic.RateLimitError("slow down", response=response(429), body=None),
            "the API refused the request: RateLimitError status=429 request_id=req_error",
        ),
        (
            lambda: anthropic.APIConnectionError(request=request()),
            "the API could not be reached: APIConnectionError",
        ),
        (
            lambda: anthropic.APIResponseValidationError(response(200), None),
            "the API's response could not be read: APIResponseValidationError",
        ),
    ],
    ids=["status", "connection", "other"],
)
def test_status_connection_and_other_api_errors_become_generation_errors(
    monkeypatch, build: Callable[[], anthropic.APIError], expected: str
):
    with pytest.raises(GenerationError) as caught:
        send_with(_raising(monkeypatch, build()))(CALL)
    assert str(caught.value) == expected
    assert caught.value.exit_code == 2
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_a_missing_credential_becomes_a_generation_error(monkeypatch):
    client = _raising(monkeypatch, TypeError(CREDENTIAL_ERROR))
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        send_with(client)(CALL)


def test_a_type_error_that_names_no_credential_propagates(monkeypatch):
    client = _raising(monkeypatch, TypeError("create() got an unexpected keyword argument 'x'"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        send_with(client)(CALL)


def test_an_echoed_key_never_reaches_the_traceback(monkeypatch):
    exc = anthropic.AuthenticationError(
        f"Error code: 401 - invalid x-api-key {SENTINEL}",
        response=response(401),
        body={"type": "error", "error": {"type": "authentication_error", "message": SENTINEL}},
    )
    assert SENTINEL in str(exc)
    with pytest.raises(GenerationError) as caught:
        send_with(_raising(monkeypatch, exc))(CALL)
    err = caught.value
    assert SENTINEL not in "".join(traceback.format_exception(err))
    assert err.__suppress_context__


def test_a_bad_request_carries_the_body_message_with_the_key_redacted(monkeypatch):
    key = "not-a-credential"
    exc = anthropic.BadRequestError(
        "Error code: 400",
        response=response(400),
        body={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": f"bad schema (key {key})"},
        },
    )
    with pytest.raises(GenerationError) as caught:
        send_with(_raising(monkeypatch, exc))(CALL)
    assert str(caught.value) == (
        "the API refused the request: BadRequestError status=400 request_id=req_error"
        ": bad schema (key [redacted])"
    )
    assert key not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__


def test_a_bad_request_without_a_readable_body_keeps_the_short_text(monkeypatch):
    exc = anthropic.BadRequestError("Error code: 400", response=response(400), body=None)
    with pytest.raises(GenerationError) as caught:
        send_with(_raising(monkeypatch, exc))(CALL)
    assert str(caught.value).endswith("status=400 request_id=req_error")


def test_the_default_send_builds_a_client_that_waits_out_an_outage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-credential")
    built: list[tuple[anthropic.Anthropic, Attempts | None]] = []
    real = api_module.send_with

    def capture(client: anthropic.Anthropic, *, attempts: Attempts | None = None) -> Send:
        built.append((client, attempts))
        return real(client, attempts=attempts)

    monkeypatch.setattr(api_module, "send_with", capture)
    assert callable(api_module.default_send())
    [(client, probe)] = built
    assert client.max_retries == 120
    assert client.max_retries > anthropic.DEFAULT_MAX_RETRIES
    assert client.timeout == anthropic.Timeout(600.0, connect=30.0)
    assert isinstance(probe, Attempts)


def test_a_bad_request_redacts_an_auth_token_too(monkeypatch):
    token = "not-a-bearer-token"
    client = anthropic.Anthropic(auth_token=token, max_retries=0)
    exc = anthropic.BadRequestError(
        "Error code: 400",
        response=response(400),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": token}},
    )

    def create(**_: object) -> anthropic.types.Message:
        raise exc

    monkeypatch.setattr(client.messages, "create", create)
    with pytest.raises(GenerationError) as caught:
        send_with(client)(CALL)
    assert str(caught.value).endswith(": [redacted]")
    assert token not in "".join(traceback.format_exception(caught.value))


def test_attempts_counts_each_http_request():
    probe = Attempts()
    request = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=2))

    sent = SimpleNamespace(status_code=200, headers={})

    def call_next(_: anthropic.APIRequest) -> SimpleNamespace:
        return sent

    assert probe.handle(request, call_next) is sent  # ty: ignore[invalid-argument-type]
    assert probe.count == 3


def test_handle_logs_a_retry_line_for_each_retryable_status(caplog):
    probe = Attempts()
    replies = iter(
        [
            SimpleNamespace(status_code=429, headers={"retry-after": "2"}),
            SimpleNamespace(status_code=200, headers={}),
        ]
    )

    def call_next(_: anthropic.APIRequest) -> SimpleNamespace:
        return next(replies)

    caplog.set_level(logging.INFO)
    CALL_NUMBER.set(7)
    for retries in range(2):
        request = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=retries))
        probe.handle(request, call_next)  # ty: ignore[invalid-argument-type]
    assert [m for m in caplog.messages if m.startswith("retry ")] == [
        "retry call=7 attempt=1 status=429 retry_after=2"
    ]


def test_a_reply_carries_the_probes_count(monkeypatch):
    client = _client()
    probe = Attempts()

    def create(**_: object) -> anthropic.types.Message:
        probe.count = 3
        return message(records_text([]))

    monkeypatch.setattr(client.messages, "create", create)
    assert send_with(client, attempts=probe)(CALL).attempts == 3


def test_a_failure_carries_the_probes_count(monkeypatch):
    client = _client()
    probe = Attempts()

    def create(**_: object) -> anthropic.types.Message:
        probe.count = 3
        raise anthropic.APIConnectionError(request=request())

    monkeypatch.setattr(client.messages, "create", create)
    with pytest.raises(GenerationError) as caught:
        send_with(client, attempts=probe)(CALL)
    assert caught.value.attempts == 3


def test_without_a_probe_attempts_is_none(monkeypatch):
    client = _client()

    def create(**_: object) -> anthropic.types.Message:
        return message(records_text([]))

    monkeypatch.setattr(client.messages, "create", create)
    assert send_with(client)(CALL).attempts is None
    assert _raising_error(monkeypatch).attempts is None


def _raising_error(monkeypatch: pytest.MonkeyPatch) -> GenerationError:
    client = _raising(monkeypatch, anthropic.APIConnectionError(request=request()))
    with pytest.raises(GenerationError) as caught:
        send_with(client)(CALL)
    return caught.value


def _async_client() -> anthropic.AsyncAnthropic:
    """A real async client with a dummy key; no test sends through its transport."""
    return anthropic.AsyncAnthropic(api_key="not-a-credential", max_retries=0)


def _status(code: int, retry_after: str | None = None) -> SimpleNamespace:
    """A stand-in for the SDK's response to one HTTP attempt."""
    headers = {} if retry_after is None else {"retry-after": retry_after}
    return SimpleNamespace(status_code=code, headers=headers)


def _attempting(
    monkeypatch: pytest.MonkeyPatch,
    client: anthropic.AsyncAnthropic,
    statuses: dict[str, list[int]],
    *,
    fail: frozenset[str] = frozenset(),
) -> None:
    """Make `client.messages.create` run each call's attempts through `Attempts.handle_async`.

    A call is named by its user turn; it takes one attempt per status in `statuses`, yielding to
    the loop between them, then returns a message, or raises a connection error when in `fail`.
    """
    probe = Attempts()

    async def create(**body: object) -> anthropic.types.Message:
        messages = cast(list[dict[str, str]], body["messages"])
        name = messages[0]["content"]
        for retries, code in enumerate(statuses[name]):

            async def call_next(_: anthropic.APIRequest, code: int = code) -> SimpleNamespace:
                await asyncio.sleep(0.001)
                return _status(code)

            attempt = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=retries))
            await probe.handle_async(attempt, call_next)  # ty: ignore[invalid-argument-type]
        if name in fail:
            raise anthropic.APIConnectionError(request=request())
        return message(records_text([]))

    monkeypatch.setattr(client.messages, "create", create)


def test_handle_async_counts_attempts_per_call(monkeypatch):
    client = _async_client()
    _attempting(monkeypatch, client, {"the user": [429, 200]})
    reply = asyncio.run(async_send_with(client)(CALL))
    assert reply.attempts == 2


def test_concurrent_calls_count_their_own_attempts(monkeypatch):
    client = _async_client()
    statuses = {"a": [429, 429, 503, 200], "b": [200], "c": [429, 500]}
    _attempting(monkeypatch, client, statuses, fail=frozenset({"c"}))
    send = async_send_with(client)

    async def both() -> list[object]:
        calls = [dataclasses.replace(CALL, user=name) for name in statuses]
        return await asyncio.gather(*(send(call) for call in calls), return_exceptions=True)

    a, b, c = asyncio.run(both())
    assert [getattr(a, "attempts", a), getattr(b, "attempts", b)] == [4, 1]
    assert isinstance(c, GenerationError)
    assert c.attempts == 2


def test_a_retry_logs_its_call_attempt_and_status(caplog):
    probe = Attempts()
    replies = iter([_status(429, "2"), _status(503), _status(400), _status(200)])

    async def call_next(_: anthropic.APIRequest) -> SimpleNamespace:
        return next(replies)

    async def attempts() -> None:
        CALL_NUMBER.set(7)
        for retries in range(4):
            request = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=retries))
            await probe.handle_async(request, call_next)  # ty: ignore[invalid-argument-type]

    caplog.set_level(logging.INFO)
    asyncio.run(attempts())
    assert [m for m in caplog.messages if m.startswith("retry ")] == [
        "retry call=7 attempt=1 status=429 retry_after=2",
        "retry call=7 attempt=2 status=503 retry_after=none",
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_the_last_permitted_attempt_logs_no_retry_line(caplog, asynchronous):
    probe = Attempts(max_retries=2)

    def call_next(_: anthropic.APIRequest) -> SimpleNamespace:
        return _status(429)

    async def call_next_async(request: anthropic.APIRequest) -> SimpleNamespace:
        return call_next(request)

    async def attempts() -> None:
        for retries in range(3):
            request = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=retries))
            if asynchronous:
                await probe.handle_async(request, call_next_async)  # ty: ignore[invalid-argument-type]
            else:
                probe.handle(request, call_next)  # ty: ignore[invalid-argument-type]

    caplog.set_level(logging.INFO)
    token = CALL_NUMBER.set(7)
    try:
        asyncio.run(attempts())
    finally:
        CALL_NUMBER.reset(token)
    assert [m for m in caplog.messages if m.startswith("retry ")] == [
        "retry call=7 attempt=1 status=429 retry_after=none",
        "retry call=7 attempt=2 status=429 retry_after=none",
    ]


def test_every_client_counts_attempts_against_its_own_retries():
    assert Attempts().max_retries == MAX_RETRIES


@pytest.mark.parametrize(("header", "logged"), [("true", True), ("false", False)])
def test_a_retry_line_follows_the_servers_should_retry_header(caplog, header, logged):
    probe = Attempts()
    status = SimpleNamespace(status_code=400 if logged else 429, headers={"x-should-retry": header})

    async def call_next(_: anthropic.APIRequest) -> SimpleNamespace:
        return status

    request = cast(anthropic.APIRequest, SimpleNamespace(retries_taken=0))
    caplog.set_level(logging.INFO)
    asyncio.run(probe.handle_async(request, call_next))  # ty: ignore[invalid-argument-type]
    assert any(m.startswith("retry ") for m in caplog.messages) is logged
