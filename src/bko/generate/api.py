"""The Anthropic SDK adapter: the only module that imports `anthropic`.

`send_with` makes one non-streaming Messages call with the SDK's default retries and reads the
message into a `Reply`. A failure with no response becomes `GenerationError`, built from the
error's type, status and request id only and raised `from None`, because an error body can echo
the credential. A 400 (`BadRequestError`) also carries the body's `error.message`, with the
client's API key and auth token replaced by `[redacted]`, so a rejected schema or parameter names
its reason.
"""

from __future__ import annotations

import anthropic

from bko.generate.errors import GenerationError
from bko.generate.messages import Call, Reply, Send, Usage, record_schema


def reply_of(message: anthropic.types.Message) -> Reply:
    """Read an SDK message into a `Reply`.

    Args:
        message: A Messages API response.

    Returns:
        The reply: its text blocks joined (thinking blocks ignored), or no text when none is there.
    """
    texts = [
        block.text for block in message.content if isinstance(block, anthropic.types.TextBlock)
    ]
    usage = message.usage
    return Reply(
        request_id=message._request_id,
        model=message.model,
        stop_reason=message.stop_reason,
        text="".join(texts) if texts else None,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        ),
        raw=message.to_dict(mode="json"),
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


def send_with(client: anthropic.Anthropic) -> Send:
    """Return a `Send` that makes each call through `client`.

    Args:
        client: The SDK client; its credential comes from the SDK's own resolution.

    Returns:
        The send, which raises `GenerationError` when no response arrives.
    """

    def send(call: Call) -> Reply:
        try:
            message = client.messages.create(
                model=call.settings.model,
                max_tokens=call.settings.max_tokens,
                system=call.system,
                messages=[{"role": "user", "content": call.user}],
                output_config={
                    "effort": call.settings.effort,
                    "format": {
                        "type": "json_schema",
                        "schema": record_schema(call.sex, call.label),
                    },
                },
            )
        except anthropic.APIStatusError as exc:
            msg = (
                f"the API refused the request: {type(exc).__name__} status={exc.status_code} "
                f"request_id={exc.request_id}"
            )
            if isinstance(exc, anthropic.BadRequestError):
                msg += _reason(exc, (client.api_key, client.auth_token))
            raise GenerationError(msg) from None
        except anthropic.APIConnectionError as exc:
            msg = f"the API could not be reached: {type(exc).__name__}"
            raise GenerationError(msg) from None
        except anthropic.APIError as exc:
            msg = f"the API's response could not be read: {type(exc).__name__}"
            raise GenerationError(msg) from None
        except TypeError as exc:
            # SDK 1.13.0 builds a client with no credential and raises this from
            # `_validate_headers` at request time; its text holds no secret. Any other
            # `TypeError` is a bug here, not a missing credential.
            if "Could not resolve authentication method" not in str(exc):
                raise
            msg = f"no Anthropic credential: set ANTHROPIC_API_KEY or run ant auth login ({exc})"
            raise GenerationError(msg) from None
        return reply_of(message)

    return send


def default_send() -> Send:
    """Return a `Send` through a client built with the SDK's own credential resolution."""
    return send_with(anthropic.Anthropic())
