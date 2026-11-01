"""On-device models' GGUF weights: each pinned by size and digest, fetched and verified.

`LOCAL_MODELS` names each model `bko.generate.local` serves. `fetch` writes a model's file to
`data/models/<model>/<sha256[:16]>/<file>` and checks its size and sha256 before any use. ADR
`research/local-generator-backend` records the choice of files.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from bko.datasets.errors import DatasetError
from bko.datasets.fetch import CHUNK_BYTES, UrlOpener, open_https
from bko.generate.errors import GenerationError


logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: Final = 600.0
"""The most one read of a download may wait."""

PROGRESS_BYTES: Final = 1 << 30
"""How often `fetch` logs its progress, in bytes."""


@dataclass(frozen=True, slots=True)
class LocalModel:
    """An on-device model: its GGUF weights pinned by size and digest.

    Attributes:
        repo: The Hugging Face repository that publishes the weights.
        filename: The GGUF file in that repository.
        size: The file's exact byte count.
        sha256: The file's lowercase hex sha256.
        template: A built-in llama.cpp chat template the server uses in place of the file's own,
            or None to render the file's template with thinking off.
        thinking: Whether the model is served with its reasoning on.
        weights: The id of the model whose file this one serves, or None for its own.
    """

    repo: str
    filename: str
    size: int
    sha256: str
    template: str | None = None
    thinking: bool = False
    weights: str | None = None

    @property
    def url(self) -> str:
        """The file's download address."""
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


_GEMMA: Final = LocalModel(
    repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
    filename="gemma-4-26B-A4B-it-Q8_0.gguf",
    size=26_859_860_992,
    sha256="b5108fd13147d1c866bb595295bc9d56f5fe744d7209f18421031d0cc47009c6",
)
_NEMOTRON: Final = LocalModel(
    repo="ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
    filename="NVIDIA-Nemotron-3-Nano-30B-A3B-Q8_0.gguf",
    size=33_585_500_064,
    sha256="0d869d64174e82fcd6fc99fcc3c63449bb4064c4896da89ec8f0e3b7f2b58e02",
)

LOCAL_MODELS: Final[Mapping[str, LocalModel]] = MappingProxyType(
    {
        "gemma-4-26b-a4b-q8_0": _GEMMA,
        "gemma-4-26b-a4b-q8_0-thinking": replace(
            _GEMMA, thinking=True, weights="gemma-4-26b-a4b-q8_0"
        ),
        "nemotron-3-nano-30b-a3b-q8_0": _NEMOTRON,
        "nemotron-3-nano-30b-a3b-q8_0-thinking": replace(
            _NEMOTRON, thinking=True, weights="nemotron-3-nano-30b-a3b-q8_0"
        ),
        "olmo-3-7b-instruct-q8_0": LocalModel(
            repo="unsloth/Olmo-3-7B-Instruct-GGUF",
            filename="Olmo-3-7B-Instruct-Q8_0.gguf",
            size=7_759_252_672,
            sha256="73a9db00c542d93578892c1fafedc9fb6668db6177262d8035ecaf70748c8122",
            template="chatml",
        ),
    }
)
"""Each on-device model by its id, read from the repository's file listing on 2026-10-29: the
publisher's own GGUF repository, else `ggml-org`'s, else `unsloth`'s. Olmo 3 7B Instruct's file
template pipes an undefined `tools` into `tojson`, which llama.cpp's template parser refuses; for a
system and a user turn with no tools it renders ChatML, so the server uses its built-in `chatml`.
Olmo 3 7B Instruct has no thinking mode. A `-thinking` id serves its base's file with reasoning
on, so the two share one download."""


def model_path(model: str, data_dir: Path) -> Path:
    """Where `model`'s weights live under `data_dir`: `models/<model>/<sha16>/<file>`.

    A thinking variant's path is its base's.
    """
    pin = LOCAL_MODELS[model]
    return data_dir / "models" / (pin.weights or model) / pin.sha256[:16] / pin.filename


def _mismatch(path: Path, size: int, sha256: str, pin: LocalModel) -> None:
    if size != pin.size or sha256 != pin.sha256:
        msg = (
            f"{path}: {size} bytes, sha256 {sha256}; "
            f"the pin is {pin.size} bytes, sha256 {pin.sha256}"
        )
        raise GenerationError(msg)


def verify_model(path: Path, pin: LocalModel) -> None:
    """Check the file at `path` against `pin`'s size and sha256, reading it in chunks.

    Raises:
        GenerationError: If the file's size or sha256 differs from the pin.
        FileNotFoundError: If the file is absent.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    _mismatch(path, size, digest.hexdigest(), pin)


def fetch(model: str, data_dir: Path, *, open_url: UrlOpener = open_https) -> Path:
    """Make `model`'s weights present and verified under `data_dir`, and return their path.

    A present file is verified and used. A download streams into `<file>.part` beside it, hashed
    as it is written, refused past the pinned size, synced and renamed into place once its size
    and sha256 match the pin, and removed if they do not; progress is logged each `PROGRESS_BYTES`.

    Raises:
        GenerationError: If the weights, present or downloaded, differ from the pin, or the
            download cannot be completed.
    """
    pin = LOCAL_MODELS[model]
    path = model_path(model, data_dir)
    if path.exists():
        logger.info("verifying model=%s path=%s", model, path)
        verify_model(path, pin)
        logger.info("weights present model=%s path=%s sha256=%s", model, path, pin.sha256)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    logger.info("fetching model=%s url=%s size=%d", model, pin.url, pin.size)
    try:
        size, sha256 = _stream(open_url, pin, part, model)
        _mismatch(part, size, sha256, pin)
    except (DatasetError, OSError, http.client.HTTPException) as exc:
        part.unlink(missing_ok=True)
        msg = f"the download of {pin.url} failed: {type(exc).__name__}: {exc}"
        raise GenerationError(msg) from None
    except GenerationError:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    logger.info("fetched model=%s path=%s sha256=%s", model, path, pin.sha256)
    return path


def _stream(open_url: UrlOpener, pin: LocalModel, part: Path, model: str) -> tuple[int, str]:
    """Copy the pin's download into `part`; return its size and sha256."""
    digest = hashlib.sha256()
    size, reported, started = 0, 0, time.monotonic()
    with open_url(pin.url, TIMEOUT_SECONDS) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            size += len(chunk)
            if size > pin.size:
                msg = f"{pin.url}: more than the pinned {pin.size} bytes arrived"
                raise GenerationError(msg)
            digest.update(chunk)
            out.write(chunk)
            if size - reported >= PROGRESS_BYTES:
                reported = size
                rate = size / max(time.monotonic() - started, 1e-9) / 1e6
                logger.info(
                    "fetch model=%s bytes=%d of=%d mb_per_s=%.1f", model, size, pin.size, rate
                )
        out.flush()
        os.fsync(out.fileno())
    return size, digest.hexdigest()
