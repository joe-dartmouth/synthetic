"""The generation request and reply, as repo-owned values: no SDK type crosses this module.

A `Send` turns one `Call` into one `Reply`; `bko.generate.api` holds the only one that reaches the
Anthropic API, so every other module and test runs on a fake. The structured-output schema fixes
each record's shape and codes: each coded field is an `enum` over its codebook, the cell's `sex`
and `target` are pinned to one value each, and the whole-number fields are `integer`. Structured
outputs cannot bound a number, so plausibility ranges and `oldpeak`'s one decimal stay in the
prompt. Each call carries its frame (`bko.datasets.frames.FrameSpec`), UCI's by default. ADR
`research/record-generation-design` records why.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME


class Level(StrEnum):
    """The guidance ladder, in order: each level adds one component to the one before it."""

    CELL = "cell"
    COHERENT = "coherent"
    FINDINGS = "findings"
    ONTOLOGY = "ontology"
    PROFILE = "profile"
    SUMMARY = "summary"


LEVELS: Final = tuple(Level)
"""The levels in ladder order."""


def reaches(level: Level, step: Level) -> bool:
    """Return whether a request at `level` carries `step`'s component."""
    return LEVELS.index(level) >= LEVELS.index(step)


type Effort = Literal["low", "medium", "high", "xhigh", "max"]
"""The `output_config.effort` values the API takes."""


@dataclass(frozen=True, slots=True)
class Settings:
    """How each call is made.

    Attributes:
        model: The model id.
        effort: The effort the model spends.
        max_tokens: The output token limit of one call.
        per_call: The most records one call asks for; a larger quota is split across calls.
        guidance: The guidance level.
    """

    model: str
    effort: Effort
    max_tokens: int
    per_call: int
    guidance: Level

    def __post_init__(self) -> None:
        """Refuse a non-positive token limit or chunk.

        Raises:
            ValueError: If `max_tokens` or `per_call` is below 1.
        """
        for name, value in (("max_tokens", self.max_tokens), ("per_call", self.per_call)):
            if value < 1:
                msg = f"{name} must be at least 1, not {value}"
                raise ValueError(msg)


SETTINGS: Final = Settings("claude-haiku-5-5", "medium", 16000, 20, Level.ONTOLOGY)
"""The settings every run uses."""


@dataclass(frozen=True, slots=True)
class Usage:
    """The tokens one call was billed for.

    Attributes:
        input_tokens: Uncached input tokens.
        output_tokens: Output tokens, thinking included.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


@dataclass(frozen=True, slots=True)
class Reply:
    """One response, read out of the SDK's message.

    Attributes:
        request_id: The API's request id, when the SDK set one.
        model: The model that served the call.
        stop_reason: Why the model stopped.
        text: The text blocks joined, or None when there are none.
        usage: The billed tokens.
        raw: The whole message as JSON values.
        attempts: The HTTP requests the SDK made for the call, retries included; None when
            nothing counted them.
    """

    request_id: str | None
    model: str
    stop_reason: str | None
    text: str | None
    usage: Usage
    raw: Mapping[str, object]
    attempts: int | None = None


@dataclass(frozen=True, slots=True)
class Call:
    """One request: the settings, the cell's pinned codes and the rendered prompt.

    Attributes:
        settings: How the call is made.
        sex: The cell's `sex` code.
        label: The cell's label code.
        system: The system prompt.
        user: The user turn.
        frame: The frame whose records the call asks for.
    """

    settings: Settings
    sex: int
    label: int
    system: str
    user: str
    frame: FrameSpec = UCI_FRAME

    def params(self) -> dict[str, object]:
        """Return the request body `bko.generate.api.send_with` sends, as keyword arguments."""
        return {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": self.user}],
            "output_config": {
                "effort": self.settings.effort,
                "format": {
                    "type": "json_schema",
                    "schema": record_schema(self.sex, self.label, frame=self.frame),
                },
            },
        }


def text_of(raw: Mapping[str, object]) -> str | None:
    """A message's text blocks joined, thinking blocks ignored; None when there are none."""
    content = raw.get("content")
    blocks = content if isinstance(content, list) else []
    texts = [
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "".join(texts) if texts else None


type Send = Callable[[Call], Reply]
"""Makes one call and returns its reply, or raises `GenerationError` when none arrives."""


def record_schema(sex: int, label: int, *, frame: FrameSpec = UCI_FRAME) -> dict[str, object]:
    """The structured-output schema of one call: a `records` array of complete `frame` records.

    Args:
        sex: The cell's `sex` code, the only one a record may hold.
        label: The cell's label code, the only one a record may hold.
        frame: The frame whose columns, codes and whole-number fields the schema fixes.

    Returns:
        A JSON schema whose every object is closed and requires all of its properties.
    """
    fields: dict[str, object] = {}
    for name in frame.columns:
        if name == "sex":
            fields[name] = {"type": "integer", "enum": [sex]}
        elif name == frame.label:
            fields[name] = {"type": "integer", "enum": [label]}
        elif name in frame.categorical:
            fields[name] = {"type": "integer", "enum": sorted(frame.codebook[name])}
        elif name in frame.fractional:
            fields[name] = {"type": "number"}
        else:
            fields[name] = {"type": "integer"}
    record = {
        "type": "object",
        "properties": fields,
        "required": list(frame.columns),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": record}},
        "required": ["records"],
        "additionalProperties": False,
    }
