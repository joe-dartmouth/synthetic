"""The base model's files load only after every pinned sha256 verifies, and only as safetensors."""

from __future__ import annotations

import hashlib
from itertools import starmap
from pathlib import Path
from typing import Final

import pytest
import torch

from bko.datasets.errors import ContractError, UnavailableError
from bko.slm.weights import (
    LOAD_FLAGS,
    OLMO_3_7B,
    BaseModel,
    WeightFile,
    fetch,
    load_base,
    model_dir,
    url,
    verify_dir,
)
from tests._datasets_fixtures import FakeOpener


BODIES: Final = {"config.json": b'{"model_type": "olmo3"}', "model.safetensors": b"\x00" * 64}
REVISION: Final = "0123456789abcdef0123456789abcdef01234567"


def _file(name: str, body: bytes) -> WeightFile:
    return WeightFile(name, len(body), hashlib.sha256(body).hexdigest())


TINY: Final = BaseModel("example/Tiny-Model", REVISION, tuple(starmap(_file, BODIES.items())))


def _written(data_dir: Path, bodies: dict[str, bytes] = BODIES) -> Path:
    directory = model_dir(TINY, data_dir)
    directory.mkdir(parents=True)
    for name, body in bodies.items():
        (directory / name).write_bytes(body)
    return directory


class Spy:
    """A `from_pretrained` stand-in that records each call and returns a module."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, directory: str, **flags: object) -> torch.nn.Module:
        self.calls.append((directory, flags))
        return torch.nn.Linear(1, 1)


def _load(data_dir: Path, model: Spy, tokenizer: Spy) -> None:
    load_base(
        TINY,
        data_dir,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        model_loader=model,
        tokenizer_loader=tokenizer,
    )


def test_a_file_whose_sha256_differs_is_refused_before_the_loader_runs(tmp_path: Path) -> None:
    _written(tmp_path, {**BODIES, "model.safetensors": b"\x01" * 64})
    model, tokenizer = Spy(), Spy()
    with pytest.raises(ContractError, match=r"model\.safetensors"):
        _load(tmp_path, model, tokenizer)
    assert model.calls == []
    assert tokenizer.calls == []


def test_a_file_outside_the_table_is_refused(tmp_path: Path) -> None:
    _written(tmp_path, {**BODIES, "pytorch_model.bin": b"pickled"})
    with pytest.raises(ContractError, match=r"pytorch_model\.bin"):
        verify_dir(TINY, tmp_path)


@pytest.mark.parametrize("name", ["pytorch_model.bin", "weights.pt", "state.pkl"])
def test_a_pin_naming_a_pickle_file_is_refused(name: str) -> None:
    with pytest.raises(ContractError, match="pickle"):
        BaseModel("example/Tiny-Model", REVISION, (_file(name, b"x"),))


def test_a_pin_with_a_moving_revision_or_a_repeated_file_is_refused() -> None:
    with pytest.raises(ContractError, match="40-hex"):
        BaseModel("example/Tiny-Model", "main", ())
    with pytest.raises(ContractError, match="more than once"):
        BaseModel("example/Tiny-Model", REVISION, (_file("a.json", b"x"), _file("a.json", b"y")))


def test_a_missing_pinned_file_is_refused_naming_it(tmp_path: Path) -> None:
    _written(tmp_path, {"config.json": BODIES["config.json"]})
    with pytest.raises(ContractError, match=r"lacks the pinned file model\.safetensors"):
        verify_dir(TINY, tmp_path)


def test_a_missing_directory_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(UnavailableError, match="fetch"):
        verify_dir(TINY, tmp_path)


def test_every_url_names_the_revision() -> None:
    for pin in (TINY, OLMO_3_7B):
        for file in pin.files:
            address = url(pin, file)
            assert address == (
                f"https://huggingface.co/{pin.repo}/resolve/{pin.revision}/{file.name}"
            )
            assert "/resolve/main/" not in address
    assert model_dir(OLMO_3_7B, Path("data")) == Path("data/models/olmo-3-1025-7b/a81bae42db3975be")


def test_the_pinned_base_model_loads_only_safetensors() -> None:
    names = [file.name for file in OLMO_3_7B.files]
    assert [name for name in names if name.startswith("model-")] == [
        f"model-0000{part}-of-00003.safetensors" for part in (1, 2, 3)
    ]


def test_the_loader_gets_the_safetensors_only_flags(tmp_path: Path) -> None:
    directory = _written(tmp_path)
    model, tokenizer = Spy(), Spy()
    _load(tmp_path, model, tokenizer)
    assert model.calls == [(str(directory), {"dtype": torch.bfloat16, **LOAD_FLAGS})]
    assert LOAD_FLAGS == {
        "local_files_only": True,
        "use_safetensors": True,
        "trust_remote_code": False,
    }
    assert tokenizer.calls == [
        (str(directory), {"local_files_only": True, "trust_remote_code": False})
    ]


def test_fetch_verifies_and_removes_a_part_file_on_mismatch(tmp_path: Path) -> None:
    opener = FakeOpener(b"\x01" * 23)
    with pytest.raises(ContractError, match=r"config\.json"):
        fetch(TINY, tmp_path, open_url=opener)
    directory = model_dir(TINY, tmp_path)
    assert list(directory.iterdir()) == []
    assert opener.calls[0][0] == url(TINY, TINY.files[0])


def test_fetch_writes_each_file_verified_and_keeps_a_present_one(tmp_path: Path) -> None:
    directory = model_dir(TINY, tmp_path)
    directory.mkdir(parents=True)
    (directory / "config.json").write_bytes(BODIES["config.json"])
    opener = FakeOpener(BODIES["model.safetensors"])
    assert fetch(TINY, tmp_path, open_url=opener) == directory
    assert [call[0] for call in opener.calls] == [url(TINY, TINY.files[1])]
    assert (directory / "model.safetensors").read_bytes() == BODIES["model.safetensors"]
    assert not list(directory.glob("*.part"))


def test_fetch_refuses_a_download_past_its_size_and_a_failed_one(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="more than the pinned"):
        fetch(TINY, tmp_path, open_url=FakeOpener(b"x" * 99))
    with pytest.raises(UnavailableError, match="failed"):
        fetch(TINY, tmp_path, open_url=FakeOpener(error=OSError("no route")))
    assert list(model_dir(TINY, tmp_path).iterdir()) == []
