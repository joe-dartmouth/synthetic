"""The base model's files at one revision, each pinned by size and sha256, fetched and verified.

`OLMO_3_7B` names every file the classifier loads. `fetch` writes them to
`data/models/olmo-3-1025-7b/<revision[:16]>/`, and `load_base` loads only after `verify_dir` has
checked every file, only as safetensors and with no remote code (ADR `tooling/slm-extra`).
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bko.datasets.errors import ContractError, DatasetError, UnavailableError
from bko.datasets.fetch import CHUNK_BYTES, UrlOpener, open_https


logger = logging.getLogger(__name__)

TIMEOUT_SECONDS: Final = 600.0
"""The most one read of a download may wait."""

PICKLE_SUFFIXES: Final = frozenset({".bin", ".pt", ".pth", ".pkl", ".ckpt"})
"""Suffixes of files that can run code on load; no pin may name one."""

LOAD_FLAGS: Final = {"local_files_only": True, "use_safetensors": True, "trust_remote_code": False}
"""The flags every weight load passes: no network, safetensors only, no remote code."""

TOKENIZER_FLAGS: Final = {"local_files_only": True, "trust_remote_code": False}
"""The flags every tokenizer load passes."""

_REVISION: Final = re.compile(r"[0-9a-f]{40}")

type Loader = Callable[..., Any]
"""A `from_pretrained` callable: a directory first, then keyword flags."""


@dataclass(frozen=True, slots=True)
class WeightFile:
    """One file of a base model.

    Attributes:
        name: The file's name in the repository and on disk.
        size: Its exact byte count.
        sha256: Its lowercase hex sha256.
    """

    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BaseModel:
    """A base model's repository, revision and every file loaded from it.

    Attributes:
        repo: The Hugging Face repository.
        revision: The commit every file is read at.
        files: The files, each pinned.

    Raises:
        ContractError: If the revision is not 40 lowercase hex digits, a file name repeats, or a
            file is a pickle format.
    """

    repo: str
    revision: str
    files: tuple[WeightFile, ...]

    def __post_init__(self) -> None:
        """Refuse a moving revision, a repeated file and a file that can run code on load."""
        if _REVISION.fullmatch(self.revision) is None:
            msg = f"{self.repo}: revision {self.revision!r} is not a 40-hex commit"
            raise ContractError(msg)
        names = [file.name for file in self.files]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            msg = f"{self.repo}: files {repeated} are named more than once"
            raise ContractError(msg)
        pickles = sorted(name for name in names if Path(name).suffix in PICKLE_SUFFIXES)
        if pickles:
            msg = f"{self.repo}: files {pickles} are pickle formats; only safetensors are loaded"
            raise ContractError(msg)


OLMO_3_7B: Final = BaseModel(
    repo="allenai/Olmo-3-1025-7B",
    revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",
    files=(
        WeightFile(
            "config.json",
            1620,
            "277d7ea5c6159d4c2d6986a8a23757d1ef06acaa7abc2c40e0513446f7db0f85",
        ),
        WeightFile(
            "generation_config.json",
            69,
            "52e8c2bef1d50a9aa99ce1da2a586ec40f524c1eded9c9ece289f3146b909ded",
        ),
        WeightFile(
            "merges.txt",
            916646,
            "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
        ),
        WeightFile(
            "model-00001-of-00003.safetensors",
            4969984976,
            "0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2",
        ),
        WeightFile(
            "model-00002-of-00003.safetensors",
            4981161496,
            "e127ea479fb6e208fe9d48d23b11212b5722f4873f6eef9c009b7a855866c641",
        ),
        WeightFile(
            "model-00003-of-00003.safetensors",
            4644917240,
            "f3ddff10052ffe5de5c6b4cad45c422c0d898acc6beb21b1b8531244adfb3c70",
        ),
        WeightFile(
            "model.safetensors.index.json",
            29630,
            "47a8266860f2a5f95c021fb4d7d3a469a95ee8b445742becbb6b19632cfc6e95",
        ),
        WeightFile(
            "special_tokens_map.json",
            207,
            "4cb2b52960bababa8ec27153831e405ef7811e3fad1226106e09c28e715cfb21",
        ),
        WeightFile(
            "tokenizer_config.json",
            4308,
            "ffa5b2cfee6e73d5dc0a7d3b7e02eb6ed60a491dd782f2bd8ad2b4659b2aafc6",
        ),
        WeightFile(
            "tokenizer.json",
            7137177,
            "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
        ),
        WeightFile(
            "vocab.json",
            1611056,
            "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195",
        ),
    ),
)
"""OLMo 3 7B base at one revision, its files read from the repository on 2026-11-24 (ADR
`tooling/slm-extra`)."""


def url(pin: BaseModel, file: WeightFile) -> str:
    """The file's download address at the pinned revision."""
    return f"https://huggingface.co/{pin.repo}/resolve/{pin.revision}/{file.name}"


def model_dir(pin: BaseModel, data_dir: Path) -> Path:
    """Where the model's files live: `models/<repo name, lowercased>/<revision[:16]>/`."""
    return data_dir / "models" / pin.repo.rsplit("/", 1)[-1].lower() / pin.revision[:16]


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _check(path: Path, file: WeightFile, size: int, sha256: str) -> None:
    if size != file.size or sha256 != file.sha256:
        msg = (
            f"{path}: {size} bytes, sha256 {sha256}; "
            f"the pin is {file.size} bytes, sha256 {file.sha256}"
        )
        raise ContractError(msg)


def verify_dir(pin: BaseModel, data_dir: Path) -> Path:
    """Check every pinned file's size and sha256 in the model's directory, and return it.

    Raises:
        UnavailableError: If the directory is absent.
        ContractError: If a pinned file is missing or differs from its pin, or the directory
            holds a file the pin does not name.
    """
    directory = model_dir(pin, data_dir)
    if not directory.is_dir():
        msg = f"{directory} is absent; run `python -m bko.slm fetch`"
        raise UnavailableError(msg)
    pinned = {file.name for file in pin.files}
    stray = sorted(path.name for path in directory.iterdir() if path.name not in pinned)
    if stray:
        msg = f"{directory} holds {stray}, which the pin of {pin.repo} does not name"
        raise ContractError(msg)
    for file in pin.files:
        path = directory / file.name
        if not path.is_file():
            msg = f"{directory} lacks the pinned file {file.name}"
            raise ContractError(msg)
        _check(path, file, *_digest(path))
    return directory


def _stream(open_url: UrlOpener, address: str, file: WeightFile, part: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open_url(address, TIMEOUT_SECONDS) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            size += len(chunk)
            if size > file.size:
                msg = f"{address}: more than the pinned {file.size} bytes arrived"
                raise ContractError(msg)
            digest.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    return size, digest.hexdigest()


def fetch(pin: BaseModel, data_dir: Path, *, open_url: UrlOpener = open_https) -> Path:
    """Make every pinned file present and verified, and return the model's directory.

    A present file is verified. A missing one streams into `<name>.part`, hashed as written,
    and is renamed into place only once its size and sha256 match; a mismatch removes the part.

    Raises:
        ContractError: If a file, present or downloaded, differs from its pin.
        UnavailableError: If a download cannot be completed.
    """
    directory = model_dir(pin, data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for file in pin.files:
        path = directory / file.name
        if path.exists():
            _check(path, file, *_digest(path))
            continue
        part = path.with_name(path.name + ".part")
        address = url(pin, file)
        logger.info("fetching file=%s url=%s size=%d", file.name, address, file.size)
        try:
            _check(part, file, *_stream(open_url, address, file, part))
        except ContractError:
            part.unlink(missing_ok=True)
            raise
        except (DatasetError, OSError, http.client.HTTPException) as exc:
            part.unlink(missing_ok=True)
            msg = f"the download of {address} failed: {type(exc).__name__}: {exc}"
            raise UnavailableError(msg) from None
        part.replace(path)
        logger.info("fetched file=%s sha256=%s", file.name, file.sha256)
    return verify_dir(pin, data_dir)


def load_base(
    pin: BaseModel,
    data_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    model_loader: Loader = AutoModelForCausalLM.from_pretrained,
    tokenizer_loader: Loader = AutoTokenizer.from_pretrained,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Verify the model's files, then load the model on `device` and its tokenizer.

    Raises:
        UnavailableError: If the directory is absent.
        ContractError: As `verify_dir` does.
    """
    directory = verify_dir(pin, data_dir)
    logger.info("weights model=%s revision=%s files=%d", pin.repo, pin.revision, len(pin.files))
    model = model_loader(str(directory), dtype=dtype, **LOAD_FLAGS)
    tokenizer = tokenizer_loader(str(directory), **TOKENIZER_FLAGS)
    model.to(device)
    return model, tokenizer
