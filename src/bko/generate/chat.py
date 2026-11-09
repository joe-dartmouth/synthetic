"""A llama.cpp server's chat-completions response as a `Reply`.

`bko.generate.local` reads each local reply with `reply_of`; ADR `research/local-generator-backend`
records the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.generate.errors import GenerationError
from bko.generate.messages import Reply, Usage, text_of


STOP_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {"stop": "end_turn", "length": "max_tokens"}
)
"""The server's `finish_reason` as the Messages API's `stop_reason`."""


def _count(counts: Mapping[str, object], key: str) -> int:
    value = counts.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def reply_of(body: object) -> Reply:
    """A chat-completions response as a `Reply`: its text, tokens and stop reason.

    A local model bills nothing, so no input is cached; the prompt's tokens are input tokens.

    Raises:
        GenerationError: Not retryable, if `body` is not a JSON object.
    """
    if not isinstance(body, dict):
        msg = f"the local server's response is not a JSON object: {type(body).__name__}"
        raise GenerationError(msg)
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    first = first if isinstance(first, dict) else {}
    finish = first.get("finish_reason")
    usage = body.get("usage")
    counts = usage if isinstance(usage, dict) else {}
    request_id = body.get("id")
    return Reply(
        request_id=request_id if isinstance(request_id, str) else None,
        model=str(body.get("model", "")),
        stop_reason=STOP_REASONS.get(finish, finish) if isinstance(finish, str) else None,
        text=text_of(body),
        usage=Usage(
            input_tokens=_count(counts, "prompt_tokens"),
            output_tokens=_count(counts, "completion_tokens"),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        raw=dict(body),
    )
