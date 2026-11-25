"""A tiny tokenizer, a tiny causal model and Diabetes records for the `bko.slm` tests.

Everything is built in memory: the tokenizer is a `tokenizers` word-level model over the words the
fixture records serialize to, and the model is a randomly initialised OLMo 3 of one layer, so no
test reads a weight file or the network. `transformers` is imported here, at module top, because
it configures logging on import (`tests/conftest.py`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd
import torch
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, Whitespace
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from bko.ontology.frames import DIABETES_FRAME
from bko.slm.serialize import ANSWERS, prompts


FRAME: Final = DIABETES_FRAME

UNKNOWN: Final = "[UNK]"
PAD: Final = "[PAD]"


def records(count: int, *, seed: int = 0) -> pd.DataFrame:
    """`count` Diabetes records, every value inside its range or codebook, labels alternating."""
    rng = np.random.default_rng(seed)
    bounds = {r.field: (int(r.low), int(r.high)) for r in FRAME.ranges}
    columns: dict[str, list[int]] = {}
    for name in FRAME.features:
        if name in FRAME.codebook:
            codes = sorted(FRAME.codebook[name])
            columns[name] = [int(rng.choice(codes)) for _ in range(count)]
        else:
            low, high = bounds[name]
            columns[name] = [int(rng.integers(low, high + 1)) for _ in range(count)]
    columns[FRAME.label] = [index % 2 for index in range(count)]
    return pd.DataFrame(columns, index=pd.RangeIndex(100, 100 + count))


def tokenizer(texts: Iterable[str]) -> PreTrainedTokenizerFast:
    """A word-level tokenizer whose vocabulary holds every word of `texts` and both answers."""
    pre = Whitespace()
    words: dict[str, int] = {UNKNOWN: 0, PAD: 1}
    for text in [*texts, *ANSWERS.values()]:
        for word, _ in pre.pre_tokenize_str(text):
            words.setdefault(word, len(words))
    model = Tokenizer(WordLevel(words, unk_token=UNKNOWN))
    model.pre_tokenizer = pre
    return PreTrainedTokenizerFast(tokenizer_object=model, unk_token=UNKNOWN, pad_token=PAD)


def character_tokenizer() -> PreTrainedTokenizerFast:
    """A tokenizer that gives every character its own token, so each answer is several."""
    characters = sorted({character for text in ANSWERS.values() for character in text})
    words = {UNKNOWN: 0, PAD: 1, **{c: index + 2 for index, c in enumerate(characters)}}
    model = Tokenizer(WordLevel(words, unk_token=UNKNOWN))
    model.pre_tokenizer = Split(Regex("."), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=model, unk_token=UNKNOWN, pad_token=PAD)


def frame_tokenizer(frame_records: pd.DataFrame) -> PreTrainedTokenizerFast:
    """`tokenizer` over the prompts of `frame_records`."""
    return tokenizer(prompts(frame_records, FRAME))


def model(vocabulary: int, *, seed: int = 0) -> PreTrainedModel:
    """A randomly initialised one-layer OLMo 3 over `vocabulary` tokens, on the CPU."""
    config = AutoConfig.for_model(
        "olmo3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=vocabulary,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AutoModelForCausalLM.from_config(config)
