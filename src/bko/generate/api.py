"""The Anthropic SDK adapter: the only module that imports `anthropic`.

`send_with` makes one non-streaming Messages call and reads the message into a `Reply`;
`default_send` builds its client with `MAX_RETRIES` retries and `TIMEOUT`. A failure with no
response becomes `GenerationError`, built from the error's type, status and request id only and
raised `from None`, because an error body can echo the credential. A 400 (`BadRequestError`) also
carries the body's `error.message`, with the client's API key and auth token replaced by
`[redacted]`, so a rejected schema or parameter names its reason.

`Attempts` is SDK middleware that counts the HTTP requests behind one call (the first, then each
retry) and notes a retried 429; `send_with` reads it into `Reply.attempts` and
`GenerationError.attempts`, and into each one's `rate_limited`, and `default_send` registers it.
A `GenerationError` is `retryable` when the API caused it: a connection error, a timeout, a status
the SDK retries below 500 (`RETRIED`: 408, 409, 429) or a status of 500 or above
(`RETRYABLE_STATUS`), so a supervised run resumes it.

`async_send_with` and `default_async_send` are the same over `AsyncAnthropic`, whose calls
`bko.generate.flight` keeps in flight together. Each such call counts its own attempts in the
`Tries` `bko.generate.messages.ATTEMPTS` holds, a `ContextVar` the flight or the send sets, so calls
in flight together never share a count, and
`Attempts.handle` and `Attempts.handle_async` log each retryable status as a `retry` line naming
the call (`bko.generate.messages.CALL_NUMBER`), its attempt, the status and the server's
`retry-after`. The SDK's own retry records, which a connection error or a timeout also produces,
name the call through `CallTag`, which `install_call_tag` adds to the SDK's client logger.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final, override

import anthropic

from bko.generate.errors import GenerationError
from bko.generate.messages import (
    ATTEMPTS,
    CALL_NUMBER,
    AsyncSend,
    Call,
    Reply,
    Send,
    Tries,
    Usage,
    text_of,
)


logger = logging.getLogger(__name__)

SDK_CLIENT_LOGGER: Final = "anthropic._base_client"
"""The logger the SDK writes its `Retrying request` records to."""

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

RETRIED: Final = frozenset({408, 409, 429})
"""The statuses below 500 the SDK retries, as its `_should_retry` does."""

RATE_LIMITED: Final = 429
"""The status of a rate-limited request."""

RETRYABLE_STATUS: Final = 500
"""The lowest status of a failure the API caused rather than the request, beside `RETRIED`."""


def retryable(status: int, headers: Mapping[str, str]) -> bool:
    """Whether the SDK retries a `status` response: per `x-should-retry`, else per its rule."""
    should = headers.get("x-should-retry")
    if should in {"true", "false"}:
        return should == "true"
    return status in RETRIED or status >= 500


def _log_retry(
    attempt: int,
    max_retries: int,
    response: anthropic.APIResponse[Any] | anthropic.AsyncAPIResponse[Any],
) -> bool:
    """Log a `retry` line when the SDK will retry `response`'s status.

    It retries only while attempts remain: attempt `max_retries + 1`, the last, raises instead.

    Returns:
        Whether the SDK retries a 429: the attempt was rate limited.
    """
    if attempt <= max_retries and retryable(response.status_code, response.headers):
        logger.warning(
            "retry call=%d attempt=%d status=%s retry_after=%s",
            CALL_NUMBER.get(),
            attempt,
            response.status_code,
            response.headers.get("retry-after") or "none",
        )
        return response.status_code == RATE_LIMITED
    return False


class Attempts(anthropic.Middleware):
    """Counts the HTTP requests the SDK makes for one call: the first, then each retry.

    Attributes:
        count: The last synchronous call's attempts.
        rate_limited: Whether the last synchronous call retried a 429.
        max_retries: The retries the client makes, so the last attempt logs no `retry` line.
    """

    def __init__(self, max_retries: int = MAX_RETRIES) -> None:
        """Start at no request, for a client of `max_retries` retries, as every client here is."""
        self.count = 0
        self.rate_limited = False
        self.max_retries = max_retries

    def handle(
        self,
        request: anthropic.APIRequest,
        call_next: Callable[[anthropic.APIRequest], anthropic.APIResponse[Any]],
    ) -> anthropic.APIResponse[Any]:
        """Record this attempt's number, make it, and log a retry.

        A status the SDK retries logs the same `retry` line as `handle_async`, and a retried 429
        marks the call rate limited.
        """
        self.count = request.retries_taken + 1
        response = call_next(request)
        if _log_retry(self.count, self.max_retries, response):
            self.rate_limited = True
        return response

    async def handle_async(
        self,
        request: anthropic.APIRequest,
        call_next: Callable[[anthropic.APIRequest], Awaitable[anthropic.AsyncAPIResponse[Any]]],
    ) -> anthropic.AsyncAPIResponse[Any]:
        """Record this attempt's number in the call's own holder, make it, and log a retry.

        The count goes to `ATTEMPTS`'s holder, not to the instance, since the async client's calls
        overlap. A status the SDK retries, with an attempt left, logs `retry call=%d attempt=%d
        status=%s retry_after=%s`, and a retried 429 marks the holder rate limited.
        """
        attempt = request.retries_taken + 1
        holder = ATTEMPTS.get()
        if holder is not None:
            holder.count = attempt
        response = await call_next(request)
        if _log_retry(attempt, self.max_retries, response) and holder is not None:
            holder.rate_limited = True
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
    *,
    rate_limited: bool = False,
) -> GenerationError:
    """The `GenerationError` a failed call raises, built from `exc`'s type, status and request id.

    It is `retryable` for a connection error (a timeout included), a status in `RETRIED` and a
    status of `RETRYABLE_STATUS` or above, and `rate_limited` when a retry was (`rate_limited`)
    or `exc` is a 429.

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
        limited = exc.status_code == RATE_LIMITED
        return GenerationError(
            msg,
            attempts=attempts,
            retryable=exc.status_code in RETRIED or exc.status_code >= RETRYABLE_STATUS,
            rate_limited=rate_limited or limited,
        )
    if isinstance(exc, anthropic.APIConnectionError):
        msg = f"the API could not be reached: {type(exc).__name__}"
        return GenerationError(msg, attempts=attempts, retryable=True, rate_limited=rate_limited)
    if isinstance(exc, anthropic.APIError):
        msg = f"the API's response could not be read: {type(exc).__name__}"
        return GenerationError(msg, attempts=attempts, rate_limited=rate_limited)
    # SDK 1.13.0 builds a client with no credential and raises this from `_validate_headers` at
    # request time; its text holds no secret. Any other `TypeError` is a bug here, not a missing
    # credential.
    if "Could not resolve authentication method" not in str(exc):
        raise exc
    msg = f"no Anthropic credential: set ANTHROPIC_API_KEY or run ant auth login ({exc})"
    return GenerationError(msg, attempts=attempts)


class CallTag(logging.Filter):
    """Appends ` call=<n>` to a record made while call n is in flight, so a retry names its call.

    `CALL_NUMBER` is a `ContextVar`, so calls in flight together each tag their own records.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Tag `record` when a call is in flight; never drop it."""
        number = CALL_NUMBER.get()
        if number > 0:
            record.msg = f"{record.msg} call={number}"
        return True


def install_call_tag() -> None:
    """Add a `CallTag` to the SDK's client logger, unless it holds one already."""
    sdk = logging.getLogger(SDK_CLIENT_LOGGER)
    if not any(isinstance(f, CallTag) for f in sdk.filters):
        sdk.addFilter(CallTag())


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

    def limited() -> bool:
        return attempts is not None and attempts.rate_limited

    def send(call: Call) -> Reply:
        if attempts is not None:
            attempts.count = 0
            attempts.rate_limited = False
        try:
            # Exactly the body the call log records, so the log is what was sent. `params` is
            # typed loosely, and the SDK's overloads want each keyword's own type.
            body: Any = call.params()
            message = client.messages.create(**body)
        except (anthropic.APIError, TypeError) as exc:
            raise _failure(exc, client, counted(), rate_limited=limited()) from None
        return dataclasses.replace(reply_of(message), attempts=counted(), rate_limited=limited())

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
        The send, whose awaited reply and failure carry the call's own attempt count and
        `rate_limited`, read from the `ATTEMPTS` holder, its caller's or its own (the count None
        when nothing counted); it raises `GenerationError` when no response arrives.
    """

    async def send(call: Call) -> Reply:
        # The caller's holder when it set one, so a call it stops still has its count.
        holder = ATTEMPTS.get()
        token = None
        if holder is None:
            holder = Tries()
            token = ATTEMPTS.set(holder)
        holder.count = 0
        holder.rate_limited = False
        try:
            body: Any = call.params()
            message = await client.messages.create(**body)
        except (anthropic.APIError, TypeError) as exc:
            raise _failure(
                exc, client, holder.count or None, rate_limited=holder.rate_limited
            ) from None
        finally:
            if token is not None:
                ATTEMPTS.reset(token)
        return dataclasses.replace(
            reply_of(message), attempts=holder.count or None, rate_limited=holder.rate_limited
        )

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
