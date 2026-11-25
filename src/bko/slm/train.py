"""A LoRA fine-tune of a causal language model as a yes-or-no classifier of serialized records.

The loss falls on the answer token alone, read at the last prompt position of each row; P(yes) is
the softmax over the two answer tokens' logits at that position (ADR
`research/slm-comparison-design`). `transformers` and `peft` are imported at module top, since
both configure logging on import.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from bko.datasets.errors import ContractError
from bko.datasets.frames import FrameSpec
from bko.slm.serialize import ANSWERS, Example, prompts


logger = logging.getLogger(__name__)

IGNORED: Final = -100
"""The answer id a scoring row carries, which no loss reads."""

ADAPTER_FILE: Final = "adapter_model.safetensors"
"""The file `save_adapter` digests: the adapter's weights."""


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """One fine-tune's LoRA and optimizer settings.

    Attributes:
        rank: LoRA rank.
        alpha: LoRA scaling numerator.
        dropout: Dropout on the LoRA input.
        learning_rate: AdamW's peak learning rate.
        warmup_share: The share of steps, rounded up, over which the rate rises linearly.
        micro_batch: Rows per optimizer step.
        epochs: Passes over the training rows.
        seed: Seeds the adapter's initialisation, its dropout and the row order.
        target_modules: The layers LoRA adapts, in `peft`'s terms.
    """

    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    warmup_share: float
    micro_batch: int
    epochs: int
    seed: int
    target_modules: str = "all-linear"


LORA: Final = LoraSettings(16, 32, 0.05, 2e-4, 0.03, 4, 1, 0)
"""The comparison's LoRA settings (ADR `research/slm-comparison-design`)."""


@dataclass(frozen=True, slots=True)
class Encoded:
    """Rows of token ids, right-padded, with each row's last prompt position.

    Attributes:
        input_ids: Token ids, one row per record, padded on the right.
        attention_mask: 1 over each row's tokens, 0 over its padding.
        last: Each row's last prompt position, where the answer is read.
        answers: Each row's answer token id, or `IGNORED` for a scoring row.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last: torch.Tensor
    answers: torch.Tensor

    def rows(self, index: Sequence[int]) -> Encoded:
        """The rows at `index`, trimmed to the widest of them."""
        chosen = torch.tensor(list(index), dtype=torch.long)
        width = int(self.attention_mask[chosen].sum(dim=1).max())
        return Encoded(
            input_ids=self.input_ids[chosen, :width],
            attention_mask=self.attention_mask[chosen, :width],
            last=self.last[chosen],
            answers=self.answers[chosen],
        )

    def to(self, device: torch.device) -> Encoded:
        """The same rows on `device`."""
        return Encoded(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            last=self.last.to(device),
            answers=self.answers.to(device),
        )


def _ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def answer_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    """The token ids of the `no` and `yes` answers.

    Raises:
        ContractError: If an answer is not exactly one token, or both are the same token.
    """
    found: list[int] = []
    for label in (0, 1):
        ids = _ids(tokenizer, ANSWERS[label])
        if len(ids) != 1:
            msg = f"answer {ANSWERS[label]!r} is {len(ids)} tokens, not one token"
            raise ContractError(msg)
        found.append(ids[0])
    if found[0] == found[1]:
        msg = f"both answers are token {found[0]}"
        raise ContractError(msg)
    return found[0], found[1]


def _pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        msg = "the tokenizer has neither a pad token nor an end token to pad with"
        raise ContractError(msg)
    return int(pad)


def _pack(rows: Sequence[list[int]], answers: Sequence[int], pad: int) -> Encoded:
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    return Encoded(
        input_ids=input_ids,
        attention_mask=attention_mask,
        last=torch.tensor([0] * len(rows), dtype=torch.long),
        answers=torch.tensor(list(answers), dtype=torch.long),
    )


def encode(tokenizer: PreTrainedTokenizerBase, examples: Sequence[Example]) -> Encoded:
    """Training rows: each prompt's ids, then its answer's id, right-padded.

    Raises:
        ContractError: If a row's ids are not its prompt's ids plus one answer id.
    """
    allowed = answer_ids(tokenizer)
    rows: list[list[int]] = []
    answers: list[int] = []
    lasts: list[int] = []
    for number, example in enumerate(examples):
        head = _ids(tokenizer, example.prompt)
        full = _ids(tokenizer, example.text)
        if len(full) != len(head) + 1 or full[:-1] != head or full[-1] not in allowed:
            msg = (
                f"example {number}: its ids are not the prompt's ids plus one answer id "
                f"({len(head)} prompt ids, {len(full)} in all)"
            )
            raise ContractError(msg)
        rows.append(full)
        answers.append(full[-1])
        lasts.append(len(head) - 1)
    packed = _pack(rows, answers, _pad_id(tokenizer))
    return Encoded(
        input_ids=packed.input_ids,
        attention_mask=packed.attention_mask,
        last=torch.tensor(lasts, dtype=torch.long),
        answers=packed.answers,
    )


def _encode_prompts(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]) -> Encoded:
    rows = [_ids(tokenizer, text) for text in texts]
    packed = _pack(rows, [IGNORED] * len(rows), _pad_id(tokenizer))
    return Encoded(
        input_ids=packed.input_ids,
        attention_mask=packed.attention_mask,
        last=torch.tensor([len(row) - 1 for row in rows], dtype=torch.long),
        answers=packed.answers,
    )


def _last_logits(logits: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(logits.shape[0], device=logits.device), last].float()


def answer_loss(logits: torch.Tensor, encoded: Encoded) -> torch.Tensor:
    """Mean cross-entropy of each row's answer token, read at its last prompt position alone."""
    return torch.nn.functional.cross_entropy(
        _last_logits(logits, encoded.last), encoded.answers.to(logits.device)
    )


def yes_probability(last_logits: torch.Tensor, ids: tuple[int, int]) -> torch.Tensor:
    """P(yes) per row: the softmax over the `no` and `yes` logits, `ids` in that order."""
    pair = last_logits[:, list(ids)].float()
    return torch.softmax(pair, dim=-1)[:, 1]


def order(count: int, seed: int) -> list[int]:
    """Row positions ordered by `sha256(f"{seed}:slm:{position}")`."""
    return sorted(
        range(count), key=lambda row: hashlib.sha256(f"{seed}:slm:{row}".encode()).hexdigest()
    )


@contextmanager
def _seeded(seed: int, device: torch.device) -> Generator[None]:
    """Seed torch's generators for the block, then restore the states they had."""
    cpu = torch.random.get_rng_state()
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def _fit(
    fitted: PeftModel,
    encoded: Encoded,
    batches: Sequence[Sequence[int]],
    *,
    settings: LoraSettings,
    device: torch.device,
) -> None:
    """Run every optimizer step of the fine-tune on `fitted`, logging each."""
    total = len(batches) * settings.epochs
    trainable = [parameter for parameter in fitted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate)
    schedule = get_linear_schedule_with_warmup(
        optimizer, math.ceil(settings.warmup_share * total), total
    )
    fitted.train()
    step = 0
    for _ in range(settings.epochs):
        for batch in batches:
            started = time.perf_counter()
            part = encoded.rows(batch).to(device)
            rate = schedule.get_last_lr()[0]
            logits = fitted(input_ids=part.input_ids, attention_mask=part.attention_mask).logits
            loss = answer_loss(logits, part)
            loss.backward()
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            logger.info(
                "step=%d of=%d loss=%.6f lr=%.3e seconds=%.3f",
                step,
                total,
                float(loss.detach()),
                rate,
                time.perf_counter() - started,
            )
    fitted.eval()


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
    *,
    settings: LoraSettings = LORA,
    device: torch.device,
) -> PeftModel:
    """Fit a LoRA adapter on `model` to answer each example's prompt with its answer.

    The base weights stay frozen. Rows run in `order`, `settings.micro_batch` per AdamW step;
    the rate rises linearly over the warmup steps, then falls linearly to zero.

    Raises:
        ContractError: If there is no example, or a row is not its prompt plus one answer id.
    """
    if not examples:
        msg = "no example to train on"
        raise ContractError(msg)
    encoded = encode(tokenizer, examples)
    rows = order(len(examples), settings.seed)
    batches = [
        rows[start : start + settings.micro_batch]
        for start in range(0, len(rows), settings.micro_batch)
    ]
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=settings.target_modules,
        task_type="CAUSAL_LM",
    )
    with _seeded(settings.seed, device):
        # `PreTrainedModel.to` is wrapped in a decorator ty reads as taking the model first.
        cast("torch.nn.Module", model).to(device)
        fitted = get_peft_model(model, config)
        _fit(fitted, encoded, batches, settings=settings, device=device)
    return fitted


def probabilities(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    records: pd.DataFrame,
    *,
    frame: FrameSpec,
    device: torch.device,
    micro_batch: int = LORA.micro_batch,
) -> pd.Series:
    """P(yes) per record, named `probability`, indexed as `records`.

    Raises:
        ContractError: As `bko.slm.serialize.prompts` and `answer_ids` do.
    """
    ids = answer_ids(tokenizer)
    if records.empty:
        return pd.Series([], index=records.index, name="probability", dtype="float64")
    encoded = _encode_prompts(tokenizer, prompts(records, frame))
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), micro_batch):
            part = encoded.rows(range(start, min(start + micro_batch, len(records)))).to(device)
            logits = model(input_ids=part.input_ids, attention_mask=part.attention_mask).logits
            values.extend(yes_probability(_last_logits(logits, part.last), ids).tolist())
    return pd.Series(values, index=records.index, name="probability", dtype="float64")


def save_adapter(model: PeftModel, path: Path) -> str:
    """Write the adapter as safetensors into `path` and return its weights file's sha256.

    Raises:
        ContractError: If `path` exists.
    """
    if path.exists():
        msg = f"{path} exists; an adapter is never written over"
        raise ContractError(msg)
    model.save_pretrained(str(path), safe_serialization=True)
    return hashlib.sha256((path / ADAPTER_FILE).read_bytes()).hexdigest()
