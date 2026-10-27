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
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Final

import anthropic

from bko.generate.errors import GenerationError
from bko.generate.messages import Call, Reply, Send, Usage, text_of


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
        except anthropic.APIStatusError as exc:
            msg = (
                f"the API refused the request: {type(exc).__name__} status={exc.status_code} "
                f"request_id={exc.request_id}"
            )
            if isinstance(exc, anthropic.BadRequestError):
                msg += _reason(exc, (client.api_key, client.auth_token))
            raise GenerationError(msg, attempts=counted()) from None
        except anthropic.APIConnectionError as exc:
            msg = f"the API could not be reached: {type(exc).__name__}"
            raise GenerationError(msg, attempts=counted()) from None
        except anthropic.APIError as exc:
            msg = f"the API's response could not be read: {type(exc).__name__}"
            raise GenerationError(msg, attempts=counted()) from None
        except TypeError as exc:
            # SDK 1.13.0 builds a client with no credential and raises this from
            # `_validate_headers` at request time; its text holds no secret. Any other
            # `TypeError` is a bug here, not a missing credential.
            if "Could not resolve authentication method" not in str(exc):
                raise
            msg = f"no Anthropic credential: set ANTHROPIC_API_KEY or run ant auth login ({exc})"
            raise GenerationError(msg, attempts=counted()) from None
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
