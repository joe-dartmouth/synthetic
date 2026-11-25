"""Records as the prompt a language-model classifier reads, and the answer it is trained on.

One template serves training and scoring: a record's prompt is one line per feature in the frame's
feature order, `<description>: <value>`, then the frame's question; its training text is the prompt
followed by the answer token for its label. A value is formatted from its column's place in the
frame, never from the dtype a row happens to carry (ADR `research/slm-comparison-design`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.errors import ContractError
from bko.datasets.frames import FrameSpec
from bko.ontology.frames import DIABETES_NAME


ANSWERS: Final[Mapping[int, str]] = MappingProxyType({0: " no", 1: " yes"})
"""Each label code's answer text; each must be one token under the base model's tokenizer."""

QUESTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {DIABETES_NAME: "Question: was this patient readmitted within 30 days of discharge? Answer:"}
)
"""The question that closes each frame's prompt, keyed by frame name."""


@dataclass(frozen=True, slots=True)
class Example:
    """One training record's prompt and the answer its label gives.

    Attributes:
        prompt: The scoring prompt.
        answer: The label's answer text, from `ANSWERS`.
    """

    prompt: str
    answer: str

    @property
    def text(self) -> str:
        """The training text: the prompt, then the answer."""
        return self.prompt + self.answer


def question(frame: FrameSpec) -> str:
    """The question that closes `frame`'s prompt.

    Raises:
        ContractError: If no question is registered for the frame.
    """
    if frame.name not in QUESTIONS:
        msg = f"frame {frame.name} has no question; QUESTIONS holds {sorted(QUESTIONS)}"
        raise ContractError(msg)
    return QUESTIONS[frame.name]


def _value(name: str, value: float, frame: FrameSpec) -> str:
    """One field's value as the prompt writes it, chosen by the field's place in `frame`."""
    if math.isnan(value):
        msg = f"field {name} holds no value; the serializer writes only complete records"
        raise ContractError(msg)
    if name in frame.codebook:
        codes = frame.codebook[name]
        if value not in codes:
            msg = f"field {name} holds code {value}, which frame {frame.name}'s codebook lacks"
            raise ContractError(msg)
        return codes[int(value)]
    if name in frame.fractional:
        return f"{value:.{frame.fraction_digits}f}"
    if not float(value).is_integer():
        msg = f"field {name} is whole in frame {frame.name} but holds {value}"
        raise ContractError(msg)
    return str(int(value))


def _join(lines: Sequence[str], frame: FrameSpec) -> str:
    return "\n".join(lines) + "\n" + question(frame)


def prompt(record: Mapping[str, float], frame: FrameSpec) -> str:
    """One record's scoring prompt.

    Raises:
        ContractError: If the frame has no question, or a value is missing, a code the codebook
            lacks, or a fraction in a whole field.
    """
    question(frame)
    lines = [
        f"{frame.descriptions[name]}: {_value(name, float(record[name]), frame)}"
        for name in frame.features
    ]
    return _join(lines, frame)


def prompts(records: pd.DataFrame, frame: FrameSpec) -> tuple[str, ...]:
    """Every record's scoring prompt, in row order, each column formatted from the frame.

    Raises:
        ContractError: As `prompt` does.
    """
    question(frame)
    columns = [
        [
            f"{frame.descriptions[name]}: {_value(name, value, frame)}"
            for value in records[name].astype("float64").tolist()
        ]
        for name in frame.features
    ]
    return tuple(_join(lines, frame) for lines in zip(*columns, strict=True))


def examples(records: pd.DataFrame, frame: FrameSpec) -> tuple[Example, ...]:
    """Every record's training example, in row order.

    Raises:
        ContractError: As `prompt` does, or if a label is not one of `ANSWERS`' codes.
    """
    texts = prompts(records, frame)
    labels = records[frame.label].tolist()
    stray = sorted({label for label in labels if label not in ANSWERS})
    if stray:
        msg = f"labels {stray} have no answer; ANSWERS holds {sorted(ANSWERS)}"
        raise ContractError(msg)
    return tuple(
        Example(prompt=text, answer=ANSWERS[int(label)])
        for text, label in zip(texts, labels, strict=True)
    )
