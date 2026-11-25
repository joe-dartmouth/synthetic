"""The LoRA trainer fits a tiny randomly initialised model on serialized records.

`transformers` and `peft` are imported at module top: both configure logging on import, and an
import inside a test body would leave the logging state changed (`tests/conftest.py`).
"""

from __future__ import annotations

import dataclasses
import math
import socket
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from transformers import PreTrainedModel, PreTrainedTokenizerFast

import bko.slm.train as train_module
from bko.datasets.errors import ContractError
from bko.evaluate.utility import Measure, Scope, utility
from bko.slm.serialize import Example, examples, prompts
from bko.slm.train import (
    LORA,
    Encoded,
    LoraSettings,
    answer_ids,
    answer_loss,
    encode,
    order,
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
from tests.conftest import NETWORK_REFUSAL


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


def test_the_comparison_lora_settings_are_pinned() -> None:
    assert (
        LoraSettings(
            rank=16,
            alpha=32,
            dropout=0.05,
            learning_rate=2e-4,
            warmup_share=0.03,
            micro_batch=4,
            epochs=1,
            seed=0,
            target_modules="all-linear",
        )
        == LORA
    )


def test_the_settings_reach_the_lora_config() -> None:
    settings = dataclasses.replace(TINY, rank=3, alpha=5, dropout=0.2)
    frame_records, words = _fixture()
    fitted = train(
        model(len(words)), words, examples(frame_records, FRAME), settings=settings, device=CPU
    )
    config = fitted.peft_config["default"]
    assert isinstance(config, LoraConfig)
    assert (config.r, config.lora_alpha, config.lora_dropout) == (3, 5, 0.2)
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert isinstance(config.target_modules, set)
    assert {name.rsplit(".", 1)[-1] for name in config.target_modules} == projections


def test_the_schedule_warms_up_over_its_share_of_every_epochs_steps(monkeypatch, caplog) -> None:
    seen: list[tuple[int, int]] = []
    real = train_module.get_linear_schedule_with_warmup

    def spy(optimizer: torch.optim.Optimizer, warmup: int, total: int):
        seen.append((warmup, total))
        return real(optimizer, warmup, total)

    monkeypatch.setattr(train_module, "get_linear_schedule_with_warmup", spy)
    caplog.set_level("INFO", logger="bko.slm.train")
    frame_records, words = _fixture()
    settings = dataclasses.replace(TINY, warmup_share=0.3, epochs=2)
    train(model(len(words)), words, examples(frame_records, FRAME), settings=settings, device=CPU)
    assert seen == [(math.ceil(0.3 * 6), 6)]
    assert "step=6 of=6 " in caplog.text


def test_the_rows_run_in_the_seeds_order_every_epoch(monkeypatch) -> None:
    taken: list[int] = []
    real = Encoded.rows

    def spy(self: Encoded, index: Sequence[int]) -> Encoded:
        taken.extend(index)
        return real(self, index)

    monkeypatch.setattr(Encoded, "rows", spy)
    frame_records, words = _fixture()
    settings = dataclasses.replace(TINY, seed=3, epochs=2)
    train(model(len(words)), words, examples(frame_records, FRAME), settings=settings, device=CPU)
    expected = order(12, 3)
    assert expected != list(range(12))
    assert taken == expected * 2


def test_a_fit_leaves_the_global_generator_as_it_found_it() -> None:
    torch.manual_seed(123)
    unfitted = torch.rand(4)
    torch.manual_seed(123)
    _fit()
    assert torch.equal(torch.rand(4), unfitted)


def _alone(fitted: PreTrainedModel | PeftModel, words, chosen: pd.DataFrame) -> list[float]:
    """Each record's P(yes) computed by hand: its prompt alone, softmax at its last token."""
    no, yes = answer_ids(words)
    values: list[float] = []
    with torch.no_grad():
        for text in prompts(chosen, FRAME):
            ids = torch.tensor([words(text, add_special_tokens=False)["input_ids"]])
            last = fitted(input_ids=ids).logits[0, -1].float()
            values.append(float(torch.softmax(last[[no, yes]], dim=-1)[1]))
    return values


@pytest.mark.parametrize("micro_batch", [1, 5])
def test_probabilities_are_each_prompts_softmax_at_its_last_token(micro_batch: int) -> None:
    frame_records, words = _fixture()
    fitted = train(
        model(len(words)), words, examples(frame_records, FRAME), settings=TINY, device=CPU
    )
    scored = probabilities(
        fitted, words, frame_records, frame=FRAME, device=CPU, micro_batch=micro_batch
    )
    assert scored.tolist() == pytest.approx(_alone(fitted, words, frame_records), abs=1e-6)


def test_probabilities_attach_to_their_records_in_any_order() -> None:
    frame_records, words = _fixture()
    fitted = train(
        model(len(words)), words, examples(frame_records, FRAME), settings=TINY, device=CPU
    )
    shuffled = [int(record) for record in np.random.default_rng(7).permutation(frame_records.index)]
    chosen = frame_records.loc[shuffled]
    scored = probabilities(fitted, words, chosen, frame=FRAME, device=CPU, micro_batch=5)
    for record in shuffled:
        alone = probabilities(fitted, words, chosen.loc[[record]], frame=FRAME, device=CPU)
        assert scored[record] == pytest.approx(alone[record], abs=1e-6)
    assert len(set(scored.round(6))) > 1


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
    assert sorted(entry.name for entry in path.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert digest == sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
    reloaded = PeftModel.from_pretrained(model(len(words)), path)
    saved, loaded = _adapter(fitted), _adapter(reloaded)
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(saved[name], loaded[name]) for name in saved)
    with pytest.raises(ContractError, match="exists"):
        save_adapter(fitted, path)


def test_the_fit_runs_with_every_socket_connection_refused() -> None:
    with pytest.raises(AssertionError, match=NETWORK_REFUSAL):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    assert _adapter(_fit())
