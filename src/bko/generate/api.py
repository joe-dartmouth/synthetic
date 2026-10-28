"""The Anthropic SDK adapter: the only module that imports `anthropic`.

`send_with` makes one non-streaming Messages call and reads the message into a `Reply`;
`default_send` builds its client with `MAX_RETRIES` retries and `TIMEOUT`. A failure with no
response becomes `GenerationError`, built from the error's type, status and request id only and
raised `from None`, because an error body can echo the credential. A 400 (`BadRequestError`) also
carries the body's `error.message`, with the client's API key and auth token replaced by
`[redacted]`, so a rejected schema or parameter names its reason.

`Attempts` is SDK middleware that counts the HTTP requests behind one call (the first, then each
retry); `send_with` reads it into `Reply.attempts` and `GenerationError.attempts`, and
`default_send` registers it.

`async_send_with` and `default_async_send` are the same over `AsyncAnthropic`, whose calls
`bko.generate.flight` keeps in flight together. Each such call counts its own attempts in a holder
it puts in `ATTEMPTS`, a `ContextVar`, so calls in flight together never share a count, and
`Attempts.handle_async` logs each retryable status as a `retry` line naming the call
(`bko.generate.messages.CALL_NUMBER`), its attempt, the status and the server's `retry-after`.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import Any, Final

import anthropic

from bko.generate.errors import GenerationError
from bko.generate.messages import CALL_NUMBER, AsyncSend, Call, Reply, Send, Usage, text_of


logger = logging.getLogger(__name__)

HTTP_LOGGERS: Final = ("httpx2",)
"""The SDK's HTTP client logs each request at INFO; the call log already holds them."""

MAX_RETRIES: Final = 120
"""HTTP retries per call. Without a server's `retry-after`, the SDK waits 0.5 s, doubling to 8 s,
each wait 0.75 to 1 times that: about 12 to 16 minutes of waits over all retries. Each attempt can
also spend up to `TIMEOUT`, so a call whose attempts all stall holds the run for hours. A stopped
run resumes only by paying its stopped call again, so one call's patience still bounds the run."""

TIMEOUT: Final = anthropic.Timeout(600.0, connect=30.0)
"""Each HTTP attempt's limits: the SDK's 600 s to read, and 30 s rather than its 5 s to connect.
Any timeout other than the SDK's default skips its refusal of a non-streaming call above 21,333
`max_tokens`; `SETTINGS.max_tokens` is 16000."""

ATTEMPTS: Final[ContextVar[list[int] | None]] = ContextVar("attempts", default=None)
"""The current async call's attempt count, in a one-item list the call made: a holder rather than
a value, so the count `Attempts.handle_async` writes reaches the call wherever the SDK runs it."""

RETRIED: Final = frozenset({408, 409, 429})
"""The statuses below 500 the SDK retries, as its `_should_retry` does."""


def retryable(status: int, headers: Mapping[str, str]) -> bool:
    """Whether the SDK retries a `status` response: per `x-should-retry`, else per its rule."""
    should = headers.get("x-should-retry")
    if should in {"true", "false"}:
        return should == "true"
    return status in RETRIED or status >= 500


class Attempts(anthropic.Middleware):
    """Counts the HTTP requests the SDK makes for one call: the first, then each retry."""

    def __init__(self) -> None:
        """Start at no request."""
        self.count = 0

    def handle(
        self,
        request: anthropic.APIRequest,
        call_next: Callable[[anthropic.APIRequest], anthropic.APIResponse[Any]],
    ) -> anthropic.APIResponse[Any]:
        """Record this attempt's number, then make it."""
        self.count = request.retries_taken + 1
        return call_next(request)

    async def handle_async(  # ruff: ignore[no-self-use]
        self,
        request: anthropic.APIRequest,
        call_next: Callable[[anthropic.APIRequest], Awaitable[anthropic.AsyncAPIResponse[Any]]],
    ) -> anthropic.AsyncAPIResponse[Any]:
        """Record this attempt's number in the call's own holder, make it, and log a retry.

        The count goes to `ATTEMPTS`'s holder, not to the instance, since the async client's calls
        overlap. A status the SDK retries logs `retry call=%d attempt=%d status=%s
        retry_after=%s`.
        """
        attempt = request.retries_taken + 1
        holder = ATTEMPTS.get()
        if holder is not None:
            holder[0] = attempt
        response = await call_next(request)
        if retryable(response.status_code, response.headers):
            logger.warning(
                "retry call=%d attempt=%d status=%s retry_after=%s",
                CALL_NUMBER.get(),
                attempt,
                response.status_code,
                response.headers.get("retry-after") or "none",
            )
        return response


def reply_of(message: anthropic.types.Message) -> Reply:
    """Read an SDK message into a `Reply`.

    Args:
        message: A Messages API response.

    Returns:
        The reply: its text blocks joined (thinking blocks ignored), or no text when none is there.
    """
    raw = message.to_dict(mode="json")
    usage = message.usage
    return Reply(
        request_id=message._request_id,
        model=message.model,
        stop_reason=message.stop_reason,
        text=text_of(raw),
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        ),
        raw=raw,
    )


def _reason(exc: anthropic.BadRequestError, secrets: tuple[str | None, ...]) -> str:
    """The 400 body's `error.message`, as `: <message>` with each secret redacted, or nothing."""
    body = exc.body
    error = body.get("error") if isinstance(body, dict) else None
    detail = error.get("message") if isinstance(error, dict) else None
    if not isinstance(detail, str) or not detail:
        return ""
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return f": {detail}"


def _failure(
    exc: anthropic.APIError | TypeError,
    client: anthropic.Anthropic | anthropic.AsyncAnthropic,
    attempts: int | None,
) -> GenerationError:
    """The `GenerationError` a failed call raises, built from `exc`'s type, status and request id.

    Raises:
        TypeError: `exc` itself, when it is not SDK 1.13.0's refusal of a missing credential.
    """
    if isinstance(exc, anthropic.APIStatusError):
        msg = (
            f"the API refused the request: {type(exc).__name__} status={exc.status_code} "
            f"request_id={exc.request_id}"
        )
        if isinstance(exc, anthropic.BadRequestError):
            msg += _reason(exc, (client.api_key, client.auth_token))
        return GenerationError(msg, attempts=attempts)
    if isinstance(exc, anthropic.APIConnectionError):
        msg = f"the API could not be reached: {type(exc).__name__}"
        return GenerationError(msg, attempts=attempts)
    if isinstance(exc, anthropic.APIError):
        msg = f"the API's response could not be read: {type(exc).__name__}"
        return GenerationError(msg, attempts=attempts)
    # SDK 1.13.0 builds a client with no credential and raises this from `_validate_headers` at
    # request time; its text holds no secret. Any other `TypeError` is a bug here, not a missing
    # credential.
    if "Could not resolve authentication method" not in str(exc):
        raise exc
    msg = f"no Anthropic credential: set ANTHROPIC_API_KEY or run ant auth login ({exc})"
    return GenerationError(msg, attempts=attempts)


def send_with(client: anthropic.Anthropic, *, attempts: Attempts | None = None) -> Send:
    """Return a `Send` that makes each call through `client`.

    Args:
        client: The SDK client; its credential comes from the SDK's own resolution.
        attempts: The middleware registered on `client`, read after each call; None counts nothing.

    Returns:
        The send, which raises `GenerationError` when no response arrives.
    """

    def counted() -> int | None:
        return (attempts.count or None) if attempts is not None else None

    def send(call: Call) -> Reply:
        if attempts is not None:
            attempts.count = 0
        try:
            # Exactly the body the call log records, so the log is what was sent. `params` is
            # typed loosely, and the SDK's overloads want each keyword's own type.
            body: Any = call.params()
            message = client.messages.create(**body)
        except (anthropic.APIError, TypeError) as exc:
            raise _failure(exc, client, counted()) from None
        return dataclasses.replace(reply_of(message), attempts=counted())

    return send


def default_send() -> Send:
    """Return a `Send` through a client built with the SDK's own credential resolution.

    The client retries `MAX_RETRIES` times within `TIMEOUT` and counts each call's HTTP attempts
    through `Attempts`.
    """
    probe = Attempts()
    client = anthropic.Anthropic(middleware=[probe], max_retries=MAX_RETRIES, timeout=TIMEOUT)
    return send_with(client, attempts=probe)


def async_send_with(client: anthropic.AsyncAnthropic) -> AsyncSend:
    """Return an `AsyncSend` that makes each call through `client`, as `send_with` does.

    Args:
        client: The async SDK client; its credential comes from the SDK's own resolution.

    Returns:
        The send, whose awaited reply and failure carry the call's own attempt count, read from
        the holder it puts in `ATTEMPTS` (None when nothing counted); it raises
        `GenerationError` when no response arrives.
    """

    async def send(call: Call) -> Reply:
        holder = [0]
        token = ATTEMPTS.set(holder)
        try:
            body: Any = call.params()
            message = await client.messages.create(**body)
        except (anthropic.APIError, TypeError) as exc:
            raise _failure(exc, client, holder[0] or None) from None
        finally:
            ATTEMPTS.reset(token)
        return dataclasses.replace(reply_of(message), attempts=holder[0] or None)

    return send


def default_async_send() -> AsyncSend:
    """Return an `AsyncSend` through a client built with the SDK's own credential resolution.

    The client retries `MAX_RETRIES` times within `TIMEOUT`, as `default_send`'s does, and
    registers `Attempts`.
    """
    client = anthropic.AsyncAnthropic(
        middleware=[Attempts()], max_retries=MAX_RETRIES, timeout=TIMEOUT
    )
    return async_send_with(client)
