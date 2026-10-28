"""Builders for the generation tests: SDK messages and errors, cells' guidance and a fake send.

The SDK's errors are built from `httpx2.Request` and `httpx2.Response`: `anthropic` depends on
`httpx2` and its error constructors take those types, so no test needs a real response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import anthropic
import httpx2
import pandas as pd

from bko.datasets.uci import UCI_FRAME
from bko.generate.errors import GenerationError
from bko.generate.guidance import Guidance, intervals
from bko.generate.messages import SETTINGS, Call, Reply, Usage
from tests._datasets_fixtures import archive, cleveland_line
from tests._validate_fixtures import BASE


def _train(chol_high: int) -> pd.DataFrame:
    """Ten UCI-shaped training records aged 35 to 70, the last one's `chol` set to `chol_high`."""
    ages = (35, 40, 44, 48, 52, 55, 58, 62, 66, 70)
    records = [
        {
            **BASE,
            "age": age,
            "sex": i % 2,
            "target": (i // 2) % 2,
            "trestbps": 110 + 5 * i,
            "chol": 180 + 10 * i,
            "thalach": 120 + 6 * i,
            "oldpeak": round(0.3 * i, 1),
        }
        for i, age in enumerate(ages)
    ]
    records[-1]["chol"] = chol_high
    return pd.DataFrame(records).astype(dict(UCI_FRAME.dtypes))


TRAIN_A: Final = _train(300)
"""A hand-built training split: ages 35 to 70, `chol` 180 to 300."""

TRAIN_B: Final = _train(400)
"""`TRAIN_A` with its `chol` maximum at 400 instead of 300."""

CELL: Final = ("<55", "female", 1)
"""A UCI cell: young women with the condition."""

GUIDANCE: Final = Guidance(
    cell=CELL, sex=0, label=1, records=2, interval=intervals(CELL, TRAIN_A, frame=UCI_FRAME)
)
"""The bare guidance of `CELL`, with two training records."""

OTHER: Final = ("<55", "male", 0)
"""A second UCI cell: young men without the condition."""

OTHER_GUIDANCE: Final = Guidance(
    cell=OTHER, sex=1, label=0, records=30, interval=intervals(OTHER, TRAIN_A, frame=UCI_FRAME)
)
"""The bare guidance of `OTHER`, with thirty training records."""

SIZES: Final = {
    (40, 0, 0): 10,
    (40, 0, 1): 0,
    (40, 1, 0): 25,
    (40, 1, 1): 25,
    (63, 0, 0): 25,
    (63, 0, 1): 25,
    (63, 1, 0): 25,
    (63, 1, 1): 25,
}
"""Records per (age, sex, num). One in five of each cell is held out, so the training split holds
twenty in every cell but two: eight young women without the condition, and no young woman with
it."""

ENDPOINT: Final = "http://127.0.0.1:9/v1/messages"
"""The URL the built requests name: the discard port, never reached."""


def message(
    text: str | None,
    *,
    stop_reason: str | None = "end_turn",
    model: str = SETTINGS.model,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    request_id: str = "req_test",
) -> anthropic.types.Message:
    """Build an SDK message: a thinking block, then a text block when `text` is not None.

    Args:
        text: The text block's text, or None for no text block.
        stop_reason: Why the model stopped.
        model: The served model.
        input_tokens: The usage's input tokens.
        output_tokens: The usage's output tokens.
        cache_creation_input_tokens: The usage's cache-write tokens; None reports none.
        cache_read_input_tokens: The usage's cache-read tokens; None reports none.
        request_id: The request id the SDK sets on a top-level response.

    Returns:
        The message.
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
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
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


def reply(
    records: list[object],
    *,
    stop_reason: str | None = "end_turn",
    model: str = SETTINGS.model,
    usage: Usage | None = None,
    request_id: str = "req_test",
) -> Reply:
    """Return a reply whose text holds `records`.

    Args:
        records: The decoded records the reply's text holds.
        stop_reason: Why the model stopped.
        model: The served model.
        usage: The billed tokens; 1000 input and 500 output when None.
        request_id: The API's request id.

    Returns:
        The reply, with a small raw message naming its id and model.
    """
    text = records_text(records)
    return Reply(
        request_id=request_id,
        model=model,
        stop_reason=stop_reason,
        text=text,
        usage=usage or Usage(1000, 500, 0, 0),
        raw={"id": "msg_test", "model": model, "content": [{"type": "text", "text": text}]},
    )


class ScriptedSend:
    """A fake `Send` that records each call and replays a script of replies and failures.

    Attributes:
        calls: Every call made, in order.
    """

    def __init__(self, *script: Reply | GenerationError) -> None:
        """Hold `script`, played one step per call."""
        self._script = list(script)
        self.calls: list[Call] = []

    def __call__(self, call: Call) -> Reply:
        """Record `call`, then return the next reply or raise the next failure."""
        self.calls.append(call)
        assert self._script, f"ScriptedSend ran out of script at call {len(self.calls)}"
        step = self._script.pop(0)
        if isinstance(step, GenerationError):
            raise step
        return step

    def assert_consumed(self) -> None:
        """Fail unless every scripted step was played."""
        assert not self._script, f"ScriptedSend has {len(self._script)} unplayed steps"


def uci(
    cp_for: Mapping[tuple[int, int], str],
    default: str,
    sizes: Mapping[tuple[int, int, int], int] = SIZES,
) -> bytes:
    """An archive of `sizes` records, chest pain `cp_for[(age, sex)]` or `default`."""
    return archive(
        "\n".join(
            cleveland_line(age=age, sex=sex, num=num, cp=cp_for.get((age, sex), default))
            for (age, sex, num), size in sizes.items()
            for _ in range(size)
        )
        + "\n"
    )
