"""Builders for the generation tests: SDK messages and errors, and one cell's guidance.

The SDK's errors are built from `httpx2.Request` and `httpx2.Response`: `anthropic` depends on
`httpx2` and its error constructors take those types, so no test needs a real response.
"""

from __future__ import annotations

import json
from typing import Final

import anthropic
import httpx2

from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import SETTINGS


CELL: Final = ("<55", "female", 1)
"""A UCI cell: young women with the condition."""

GUIDANCE: Final = Guidance(cell=CELL, sex=0, label=1, records=2, interval=intervals(CELL))
"""The bare guidance of `CELL`, with two training records."""

ENDPOINT: Final = "http://127.0.0.1:9/v1/messages"
"""The URL the built requests name: the discard port, never reached."""


def message(
    text: str | None,
    *,
    stop_reason: str | None = "end_turn",
    model: str = SETTINGS.model,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    request_id: str = "req_test",
) -> anthropic.types.Message:
    """Build an SDK message: a thinking block, then a text block when `text` is not None.

    Args:
        text: The text block's text, or None for no text block.
        stop_reason: Why the model stopped.
        model: The served model.
        input_tokens: The usage's input tokens.
        output_tokens: The usage's output tokens.
        request_id: The request id the SDK sets on a top-level response.

    Returns:
        The message, with no cache usage reported.
    """
    content: list[dict[str, object]] = [
        {"type": "thinking", "thinking": "considering the cell", "signature": "sig"}
    ]
    if text is not None:
        content.append({"type": "text", "text": text})
    built = anthropic.types.Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            },
        }
    )
    built._request_id = request_id
    return built


def records_text(records: list[object]) -> str:
    """Return the structured-output text holding `records`."""
    return json.dumps({"records": records})


def response(status: int, *, request_id: str = "req_error") -> httpx2.Response:
    """Return a response with `status` to a POST of `ENDPOINT`, carrying `request_id`."""
    return httpx2.Response(
        status,
        request=httpx2.Request("POST", ENDPOINT),
        headers={"request-id": request_id},
    )


def request() -> httpx2.Request:
    """Return a POST of `ENDPOINT`."""
    return httpx2.Request("POST", ENDPOINT)
