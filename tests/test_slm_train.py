"""The LoRA trainer fits a tiny randomly initialised model on serialized records.

`transformers` and `peft` are imported at module top: both configure logging on import, and an
import inside a test body would leave the logging state changed (`tests/conftest.py`).
"""

from __future__ import annotations

import dataclasses
import math
import socket
from pathlib import Path

import pandas as pd
import pytest
import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import PreTrainedTokenizerFast

from bko.datasets.errors import ContractError
from bko.evaluate.utility import Measure, Scope, utility
from bko.slm.serialize import ANSWERS, Example, examples
from bko.slm.train import (
    LORA,
    answer_ids,
    answer_loss,
    encode,
    probabilities,
    save_adapter,
    train,
    yes_probability,
)
from tests._slm_fixtures import (
    FRAME,
    character_tokenizer,
    frame_tokenizer,
    model,
    records,
    tokenizer,
)


CPU = torch.device("cpu")
TINY = dataclasses.replace(LORA, rank=4, alpha=8, dropout=0.1, learning_rate=1e-2)


def _fixture(count: int = 12) -> tuple[pd.DataFrame, PreTrainedTokenizerFast]:
    frame_records = records(count)
    return frame_records, frame_tokenizer(frame_records)


def _fit(*, seed: int = 0) -> PeftModel:
    frame_records, words = _fixture()
    settings = dataclasses.replace(TINY, seed=seed)
    return train(
        model(len(words)), words, examples(frame_records, FRAME), settings=settings, device=CPU
    )


def _adapter(fitted: PeftModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(fitted).items()
    }


def test_the_loss_falls_on_the_answer_token_alone() -> None:
    frame_records, words = _fixture(4)
    encoded = encode(words, examples(frame_records, FRAME))
    no, yes = answer_ids(words)
    for row in range(4):
        width = int(encoded.last[row]) + 2
        assert encoded.attention_mask[row, :width].all()
        assert not encoded.attention_mask[row, width:].any()
        assert int(encoded.input_ids[row, width - 1]) == int(encoded.answers[row])
    assert set(encoded.answers.tolist()) == {no, yes}
    logits = torch.randn(*encoded.input_ids.shape, len(words))
    expected = torch.stack(
        [
            -torch.log_softmax(logits[row, int(encoded.last[row])], dim=-1)[
                int(encoded.answers[row])
            ]
            for row in range(4)
        ]
    ).mean()
    assert torch.allclose(answer_loss(logits, encoded), expected)
    moved = logits.clone()
    for row in range(4):
        moved[row, : int(encoded.last[row])] += 5.0
        moved[row, int(encoded.last[row]) + 1 :] -= 5.0
    assert torch.allclose(answer_loss(moved, encoded), expected)


def test_a_tiny_fit_moves_the_adapter_and_leaves_the_base_weights_alone() -> None:
    frame_records, words = _fixture()
    base = model(len(words))
    before = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    fitted = train(base, words, examples(frame_records, FRAME), settings=TINY, device=CPU)
    after = fitted.get_base_model().state_dict()
    for name, tensor in before.items():
        stored = name.replace(".weight", ".base_layer.weight")
        current = after[name] if name in after else after[stored]
        assert torch.equal(current, tensor), name
    lora_b = [tensor for name, tensor in _adapter(fitted).items() if "lora_B" in name]
    assert lora_b
    assert any(bool(tensor.abs().sum() > 0) for tensor in lora_b)


def test_two_fits_at_one_seed_give_equal_adapters_on_cpu() -> None:
    first, second = _adapter(_fit()), _adapter(_fit())
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_another_seed_gives_another_adapter() -> None:
    first, other = _adapter(_fit(seed=0)), _adapter(_fit(seed=1))
    assert not all(torch.equal(first[name], other[name]) for name in first)


def test_the_yes_probability_is_the_softmax_of_the_two_answer_logits() -> None:
    logits = torch.full((1, 6), 100.0)
    logits[0, 2], logits[0, 4] = 0.0, 2.0
    assert yes_probability(logits, (2, 4)).tolist() == pytest.approx([0.8807970779778823])


def test_probabilities_are_indexed_by_the_records_and_feed_utility() -> None:
    frame_records, words = _fixture()
    fitted = train(
        model(len(words)), words, examples(frame_records, FRAME), settings=TINY, device=CPU
    )
    test = frame_records.iloc[::-1]
    scored = probabilities(fitted, words, test, frame=FRAME, device=CPU, micro_batch=5)
    assert scored.name == "probability"
    assert scored.dtype == "float64"
    assert scored.index.equals(test.index)
    assert ((scored > 0) & (scored < 1)).all()
    measures = utility(test, scored, set(), edges=FRAME.edges, calibration=True)
    assert math.isfinite(measures[Measure.BRIER, Scope.ALL])


def test_an_answer_of_two_tokens_is_refused() -> None:
    with pytest.raises(ContractError, match="one token"):
        answer_ids(character_tokenizer())
    words = tokenizer(["Answer :", "maybe"])
    with pytest.raises(ContractError, match="answer id"):
        encode(words, [Example(prompt="Answer :", answer=" maybe")])


def test_the_adapter_is_saved_as_safetensors_and_reloads_equal(tmp_path: Path) -> None:
    frame_records, words = _fixture()
    fitted = train(
        model(len(words)), words, examples(frame_records, FRAME), settings=TINY, device=CPU
    )
    path = tmp_path / "adapter"
    digest = save_adapter(fitted, path)
    assert len(digest) == 64
    assert (path / "adapter_model.safetensors").is_file()
    assert not list(path.glob("*.bin"))
    reloaded = PeftModel.from_pretrained(model(len(words)), path)
    saved, loaded = _adapter(fitted), _adapter(reloaded)
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(saved[name], loaded[name]) for name in saved)
    with pytest.raises(ContractError, match="exists"):
        save_adapter(fitted, path)


def test_the_fit_reaches_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        msg = "the fit reached the network"
        raise AssertionError(msg)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    fitted = _fit()
    assert _adapter(fitted)
    assert set(ANSWERS) == {0, 1}
