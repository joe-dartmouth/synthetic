"""`bko.generate.weights`: each local model's weights pinned, fetched and verified."""

from __future__ import annotations

import hashlib
import io

import pytest

import bko.generate.weights as weights_module
from bko.generate.errors import GenerationError
from bko.generate.weights import LOCAL_MODELS, LocalModel, fetch, model_path, verify_model


PIN_BYTES = b"weights" * 1000
PIN = LocalModel(
    repo="org/repo",
    filename="w.gguf",
    size=len(PIN_BYTES),
    sha256=hashlib.sha256(PIN_BYTES).hexdigest(),
)


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(weights_module, "LOCAL_MODELS", {"toy": PIN})
    return "toy"


def test_fetch_verifies_size_and_digest(pinned, tmp_path):
    seen: list[str] = []

    def opener(url: str, timeout: float):
        seen.append(url)
        return io.BytesIO(PIN_BYTES)

    path = fetch(pinned, tmp_path, open_url=opener)
    assert path == tmp_path / "models" / "toy" / PIN.sha256[:16] / "w.gguf"
    assert path.read_bytes() == PIN_BYTES
    assert seen == ["https://huggingface.co/org/repo/resolve/main/w.gguf"]
    assert fetch(pinned, tmp_path, open_url=opener) == path
    assert len(seen) == 1


@pytest.mark.parametrize(
    "served",
    [PIN_BYTES[:-1], b"X" + PIN_BYTES[1:]],
    ids=["size-only", "digest-only"],
)
def test_a_download_that_differs_from_the_pin_is_refused_and_removed(pinned, tmp_path, served):
    with pytest.raises(GenerationError, match="the pin is"):
        fetch(pinned, tmp_path, open_url=lambda url, timeout: io.BytesIO(served))
    path = model_path(pinned, tmp_path)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


@pytest.mark.parametrize(
    "stored", [PIN_BYTES[:-1], b"X" + PIN_BYTES[1:]], ids=["size-only", "digest-only"]
)
def test_a_present_file_differing_in_size_or_digest_alone_is_refused(tmp_path, stored):
    path = tmp_path / "w.gguf"
    path.write_bytes(stored)
    with pytest.raises(GenerationError, match="the pin is"):
        verify_model(path, PIN)


def test_a_download_past_the_pinned_size_stops(pinned, tmp_path):
    with pytest.raises(GenerationError, match="more than the pinned"):
        fetch(pinned, tmp_path, open_url=lambda url, timeout: io.BytesIO(PIN_BYTES + b"x"))
    assert not model_path(pinned, tmp_path).exists()


def test_a_failed_download_is_a_generation_error(pinned, tmp_path):
    def broken(url: str, timeout: float):
        raise OSError("network down")

    with pytest.raises(GenerationError, match="network down"):
        fetch(pinned, tmp_path, open_url=broken)


def test_a_present_file_that_differs_from_the_pin_is_refused(tmp_path):
    path = tmp_path / "w.gguf"
    path.write_bytes(PIN_BYTES + b"x")
    with pytest.raises(GenerationError, match="the pin is"):
        verify_model(path, PIN)


def test_every_pinned_model_has_a_digest_and_size():
    for model, pin in LOCAL_MODELS.items():
        assert len(pin.sha256) == 64
        assert pin.size > 0
        assert pin.filename.endswith("Q8_0.gguf"), model
