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
from http import HTTPStatus

from bko.generate.messages import Reply


logger = logging.getLogger(__name__)


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


def reasoning_tokens(text: str, connection: http.client.HTTPConnection) -> int | None:
    """The tokens the server's `/tokenize` counts in `text`, or None if it cannot say.

    Any failure, a refused connection, a refusing status or a body with no list `tokens`, reads
    as None, so a count never costs the call its reply.
    """
    body = json.dumps({"content": text, "add_special": False}).encode()
    try:
        connection.request(
            "POST", "/tokenize", body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        data = response.read()
        tokens = json.loads(data)["tokens"] if response.status == HTTPStatus.OK else None
    except OSError, http.client.HTTPException, ValueError, KeyError, TypeError:
        return None
    return len(tokens) if isinstance(tokens, list) else None


def log_reasoning(reply: Reply, connection: http.client.HTTPConnection) -> None:
    """Log the reply's `reasoning` line: its reasoning and content lengths and tokens.

    The line is a WARNING ending ` reason=truncated_while_reasoning` when the reply stopped at
    `max_tokens` with reasoning and no content, the budget spent on thinking.
    """
    text = reasoning_text(reply) or ""
    tokens = reasoning_tokens(text, connection) if text else 0
    content = len(reply.text or "")
    truncated = reply.stop_reason == "max_tokens" and not content and bool(text)
    logger.log(
        logging.WARNING if truncated else logging.INFO,
        "reasoning request_id=%s stop_reason=%s reasoning_chars=%d reasoning_tokens=%s "
        "content_chars=%d output_tokens=%d%s",
        reply.request_id,
        reply.stop_reason,
        len(text),
        "unknown" if tokens is None else tokens,
        content,
        reply.usage.output_tokens,
        " reason=truncated_while_reasoning" if truncated else "",
    )
