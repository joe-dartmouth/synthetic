"""A local thinking reply's reasoning: its length, its tokens and its per-call line.

`bko.generate.local` refuses a `-thinking` server whose pre-flight reply carries no reasoning
(`reasoning_chars`), and logs each live `-thinking` call's `reasoning` line (`log_reasoning`), its
reasoning tokens counted by the server's `/tokenize`, since the reply's `usage` holds no reasoning
count. ADR `research/local-generator-backend` records why.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
from http import HTTPStatus
from typing import Final

from bko.generate.messages import Reply


logger = logging.getLogger(__name__)


TOKENIZE_SECONDS: Final = 30.0
"""The most one `/tokenize` socket operation may wait; a slower count logs as `unknown`."""

FIELD_CHARS: Final = 80
"""The most characters of a server-sent field the `reasoning` line quotes, escapes included."""


def reasoning_text(reply: Reply) -> str | None:
    """The reply's `reasoning_content`, or None when absent or not text."""
    choices = reply.raw.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    thoughts = message.get("reasoning_content") if isinstance(message, dict) else None
    return thoughts if isinstance(thoughts, str) else None


def reasoning_chars(reply: Reply) -> int:
    """The characters of the reply's `reasoning_content`, 0 when absent or not text."""
    return len(reasoning_text(reply) or "")


def reasoning_tokens(text: str, connection: http.client.HTTPConnection) -> int | str:
    """The tokens the server's `/tokenize` counts in `text`, or why it cannot say.

    The request waits at most `TOKENIZE_SECONDS` per socket operation, then the connection's own
    timeout is restored. Any failure, a refused or timed-out connection, a refusing status or a
    body with no list `tokens`, returns its status or exception type name, so a count never costs
    the call its reply.
    """
    body = json.dumps({"content": text, "add_special": False}).encode()
    timeout = connection.timeout
    connection.timeout = TOKENIZE_SECONDS
    if connection.sock is not None:
        connection.sock.settimeout(TOKENIZE_SECONDS)
    try:
        connection.request(
            "POST", "/tokenize", body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        data = response.read()
        if response.status != HTTPStatus.OK:
            return f"status={response.status}"
        tokens = json.loads(data)["tokens"]
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
    ) as exc:
        return type(exc).__name__
    finally:
        connection.timeout = timeout
        if connection.sock is not None:
            connection.sock.settimeout(timeout)
    return len(tokens) if isinstance(tokens, list) else type(tokens).__name__


def field(value: object) -> str:
    """`value` as one bare token of the `reasoning` line, at most `FIELD_CHARS` characters.

    A whitespace or unprintable character reads as a backslash, `u` and its code point in at least
    four hex digits, so a plain id or stop reason reads as it is and a crafted one cannot split or
    forge the line; a longer field is cut and ends `...`.
    """
    text = "".join(
        c if c.isprintable() and not c.isspace() else f"\\u{ord(c):04x}" for c in str(value)
    )
    return text if len(text) <= FIELD_CHARS else f"{text[: FIELD_CHARS - 3]}..."


def log_reasoning(
    reply: Reply, connection: http.client.HTTPConnection, warned: threading.Event
) -> None:
    """Log the reply's `reasoning` line: its reasoning and content lengths and tokens.

    The line is a WARNING ending ` reason=truncated_while_reasoning` when the reply stopped at
    `max_tokens` with reasoning and no content, the budget spent on thinking. The first failed
    `/tokenize` of a run, `warned` unset, also logs one `tokenize failed` WARNING naming why.
    """
    text = reasoning_text(reply) or ""
    tokens = reasoning_tokens(text, connection) if text else 0
    if not isinstance(tokens, int) and not warned.is_set():
        warned.set()
        logger.warning("tokenize failed: %s; this call logs reasoning_tokens=unknown", tokens)
    content = len(reply.text or "")
    truncated = reply.stop_reason == "max_tokens" and not content and bool(text)
    logger.log(
        logging.WARNING if truncated else logging.INFO,
        "reasoning request_id=%s stop_reason=%s reasoning_chars=%d reasoning_tokens=%s "
        "content_chars=%d output_tokens=%d%s",
        field(reply.request_id),
        field(reply.stop_reason),
        len(text),
        tokens if isinstance(tokens, int) else "unknown",
        content,
        reply.usage.output_tokens,
        " reason=truncated_while_reasoning" if truncated else "",
    )
