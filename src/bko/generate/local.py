"""On-device generators: a `Send` over a llama.cpp server on this machine, and its weights.

`local_send` posts each `Call` to `llama-server`'s OpenAI-compatible `/v1/chat/completions` on
`127.0.0.1`, with the call's system and user turns, its `max_tokens`, sampling fixed at
temperature 1.0 and top-p 1.0 (the Messages API's default temperature), the call's
`record_schema` as a `json_schema` response format, and thinking off. Nothing leaves the machine.
A refused connection or a status of 500 or above raises a retryable `GenerationError`, as the
API's connection errors do (`bko.generate.api`).

`LOCAL_MODELS` pins each model's GGUF weights by repository, file, size and sha256. The command
fetches and verifies them, prints the server's argv, and probes a running server with one call:

    python -m bko.generate.local fetch <model> [--data-dir <dir>]
    python -m bko.generate.local serve-command <model> [--data-dir <dir>]
    python -m bko.generate.local probe <model>

ADR `research/local-generator-backend` records the backend and why.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import shlex
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from bko.datasets.fetch import CHUNK_BYTES, UrlOpener, default_data_dir, open_https
from bko.datasets.uci import UCI_FRAME
from bko.generate.decode import decode, undecoded_in
from bko.generate.errors import GenerationError
from bko.generate.guidance import Guidance
from bko.generate.messages import SETTINGS, AsyncSend, Call, Reply, Usage, record_schema
from bko.generate.prompt import render


logger = logging.getLogger(__name__)


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
    """

    repo: str
    filename: str
    size: int
    sha256: str
    template: str | None = None

    @property
    def url(self) -> str:
        """The file's download address."""
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


LOCAL_MODELS: Final[Mapping[str, LocalModel]] = MappingProxyType(
    {
        "gemma-4-26b-a4b-q8_0": LocalModel(
            repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
            filename="gemma-4-26B-A4B-it-Q8_0.gguf",
            size=26_859_860_992,
            sha256="b5108fd13147d1c866bb595295bc9d56f5fe744d7209f18421031d0cc47009c6",
        ),
        "nemotron-3-nano-30b-a3b-q8_0": LocalModel(
            repo="ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
            filename="NVIDIA-Nemotron-3-Nano-30B-A3B-Q8_0.gguf",
            size=33_585_500_064,
            sha256="0d869d64174e82fcd6fc99fcc3c63449bb4064c4896da89ec8f0e3b7f2b58e02",
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
Olmo 3 7B Instruct has no thinking mode."""

LOCAL_URL: Final = "http://127.0.0.1:8081"
"""The address `llama-server` listens on; it binds the loopback interface only."""

SLOTS: Final = 4
"""Parallel slots the server keeps, so a run keeps this many calls in flight."""

CONTEXT: Final = 81920
"""The server's context in tokens, shared by its slots: 20480 per call."""

TIMEOUT_SECONDS: Final = 1800.0
"""The most one call may take, its whole generation included."""

STOP_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {"stop": "end_turn", "length": "max_tokens"}
)
"""The server's `finish_reason` as the Messages API's `stop_reason`."""

RETRYABLE_STATUS: Final = 500
"""The lowest status that marks a server-side failure a resume may get past."""


def payload(call: Call) -> dict[str, object]:
    """The chat-completions request body for `call`."""
    return {
        "model": call.settings.model,
        "messages": [
            {"role": "system", "content": call.system},
            {"role": "user", "content": call.user},
        ],
        "max_tokens": call.settings.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "records",
                "strict": True,
                "schema": record_schema(call.sex, call.label, frame=call.frame),
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def reply_of(body: Mapping[str, object]) -> Reply:
    """A chat-completions response as a `Reply`: its text, tokens and stop reason.

    A local model bills nothing, so no input is cached; the prompt's tokens are input tokens.
    """
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    text = message.get("content") if isinstance(message, dict) else None
    finish = first.get("finish_reason") if isinstance(first, dict) else None
    usage = body.get("usage")
    counts = usage if isinstance(usage, dict) else {}
    request_id = body.get("id")
    return Reply(
        request_id=request_id if isinstance(request_id, str) else None,
        model=str(body.get("model", "")),
        stop_reason=STOP_REASONS.get(str(finish), None if finish is None else str(finish)),
        text=text if isinstance(text, str) else None,
        usage=Usage(
            input_tokens=int(counts.get("prompt_tokens", 0)),
            output_tokens=int(counts.get("completion_tokens", 0)),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        raw=dict(body),
    )


def local_send(model: str, *, url: str = LOCAL_URL) -> AsyncSend:
    """Post each call to the llama.cpp server at `url`, in a worker thread.

    Args:
        model: The model id the server serves under its alias; each call's settings name it.
        url: The server's address.

    Returns:
        A send whose replies come from the server.
    """
    if not url.startswith("http://127.0.0.1:"):
        msg = f"refusing {url}: a local model is served on the loopback interface only"
        raise ValueError(msg)
    endpoint = f"{url}/v1/chat/completions"

    def post(call: Call) -> Reply:
        if call.settings.model != model:
            msg = f"the call names model {call.settings.model}; this server serves {model}"
            raise GenerationError(msg)
        data = json.dumps(payload(call)).encode()
        # The endpoint is checked above to be a loopback http:// address.
        request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
            endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(  # ruff: ignore[suspicious-url-open-usage]
                request, timeout=TIMEOUT_SECONDS
            ) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            exc.close()
            msg = f"the local server refused the request: status={exc.code}"
            raise GenerationError(msg, retryable=exc.code >= RETRYABLE_STATUS) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            msg = f"the local server could not be reached: {type(exc).__name__}"
            raise GenerationError(msg, retryable=True) from None
        return reply_of(body)

    async def send(call: Call) -> Reply:
        return await asyncio.to_thread(post, call)

    return send


def model_path(model: str, data_dir: Path) -> Path:
    """Where `model`'s weights live under `data_dir`: `models/<model>/<sha16>/<file>`."""
    pin = LOCAL_MODELS[model]
    return data_dir / "models" / model / pin.sha256[:16] / pin.filename


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
    if size != pin.size or digest.hexdigest() != pin.sha256:
        msg = (
            f"{path}: {size} bytes, sha256 {digest.hexdigest()}; "
            f"the pin is {pin.size} bytes, sha256 {pin.sha256}"
        )
        raise GenerationError(msg)


def fetch(model: str, data_dir: Path, *, open_url: UrlOpener = open_https) -> Path:
    """Make `model`'s weights present and verified under `data_dir`, and return their path.

    A present file is verified and used. A download streams into `<file>.part` beside it and is
    renamed into place once its size and sha256 match the pin.

    Raises:
        GenerationError: If the weights, present or downloaded, differ from the pin.
    """
    pin = LOCAL_MODELS[model]
    path = model_path(model, data_dir)
    if path.exists():
        verify_model(path, pin)
        logger.info("weights present model=%s path=%s sha256=%s", model, path, pin.sha256)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    logger.info("fetching model=%s url=%s size=%d", model, pin.url, pin.size)
    with open_url(pin.url, TIMEOUT_SECONDS) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
    verify_model(part, pin)
    part.rename(path)
    logger.info("fetched model=%s path=%s sha256=%s", model, path, pin.sha256)
    return path


def serve_command(model: str, data_dir: Path) -> list[str]:
    """The `llama-server` argv that serves `model` on `LOCAL_URL` with thinking off.

    A model with a built-in `template` is served with it; any other renders its file's template.
    """
    template = LOCAL_MODELS[model].template
    rendering = (
        ["--chat-template", template]
        if template is not None
        else [
            "--jinja",
            "--chat-template-kwargs",
            json.dumps({"enable_thinking": False}, separators=(",", ":")),
        ]
    )
    return [
        "llama-server",
        "-m",
        str(model_path(model, data_dir)),
        "--alias",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        LOCAL_URL.rsplit(":", 1)[1],
        "-np",
        str(SLOTS),
        "-c",
        str(CONTEXT),
        *rendering,
    ]


PROBE_CELL: Final = ("<55", "female", 1)
"""The cell the probe asks for: E1's smallest. Its guidance bounds each measured field by UCI's
plausible range, age below the band edge."""


def probe(model: str, *, url: str = LOCAL_URL, rows: int = 4) -> int:
    """Send one call for `rows` records of `PROBE_CELL` and log what decodes.

    Returns:
        The records that decoded, of `rows`.

    Raises:
        GenerationError: If the server cannot be reached or refuses the call.
    """
    settings = dataclasses.replace(SETTINGS, model=model)
    interval = {r.field: (r.low, r.high) for r in UCI_FRAME.ranges}
    interval["age"] = (interval["age"][0], float(UCI_FRAME.edges[0] - 1))
    guidance = Guidance(cell=PROBE_CELL, sex=0, label=1, records=2, interval=interval)
    system, user = render(guidance, rows, settings.guidance, frame=UCI_FRAME)
    call = Call(settings, 0, 1, system, user, frame=UCI_FRAME)
    reply = asyncio.run(local_send(model, url=url)(call))
    candidates = decode(reply, rows)
    decoded = len(candidates) - undecoded_in(candidates)
    logger.info(
        "probe model=%s asked=%d decoded=%d stop_reason=%s input_tokens=%d output_tokens=%d",
        model,
        rows,
        decoded,
        reply.stop_reason,
        reply.usage.input_tokens,
        reply.usage.output_tokens,
    )
    for candidate in candidates:
        logger.info("probe record %s", json.dumps(candidate, default=str, sort_keys=True))
    return decoded


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return its exit status: 0 done, 2 a fetch or probe failed."""
    parser = argparse.ArgumentParser(prog="python -m bko.generate.local", description=__doc__)
    parser.add_argument("command", choices=("fetch", "serve-command", "probe"))
    parser.add_argument("model", choices=sorted(LOCAL_MODELS))
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    if args.command == "serve-command":
        sys.stdout.write(shlex.join(serve_command(args.model, data_dir)) + "\n")
        return 0
    try:
        if args.command == "fetch":
            fetch(args.model, data_dir)
            return 0
        decoded = probe(args.model)
    except GenerationError as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    if decoded == 0:
        logger.error("probe model=%s decoded no record", args.model)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
