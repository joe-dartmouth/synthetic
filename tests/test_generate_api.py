"""The SDK adapter: a message read into a `Reply`, and the SDK's failures into `GenerationError`."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import anthropic
import pytest

import bko.generate.api as api_module
from bko.generate.api import Attempts, reply_of, send_with
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS, Call, Send, Usage
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

    def call_next(_: anthropic.APIRequest) -> str:
        return "sent"

    assert probe.handle(request, call_next) == "sent"  # ty: ignore[invalid-argument-type]
    assert probe.count == 3


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
