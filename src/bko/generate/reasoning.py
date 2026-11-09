"""A local thinking reply's reasoning: its length in characters.

`bko.generate.local` refuses a `-thinking` server whose pre-flight reply carries no reasoning
(`reasoning_chars`). ADR `research/local-generator-backend` records why.
"""

from __future__ import annotations

from bko.generate.messages import Reply


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
