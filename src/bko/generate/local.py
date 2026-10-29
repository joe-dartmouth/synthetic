"""On-device generators: a `Send` over a llama.cpp server on this machine, and its weights.

`local_send` posts each `Call` to `llama-server`'s OpenAI-compatible `/v1/chat/completions` on
`127.0.0.1`, over `http.client` (no proxy, no redirect), with the call's system and user turns, its
`max_tokens`, sampling fixed at temperature 1.0 and top-p 1.0 (the Messages API's default
temperature), the call's `record_schema` as a `json_schema` response format, and thinking off.
Nothing leaves the machine. Its first call reads the server's `/props` and refuses a server whose
alias or weights file is not the model's, logging the server's build and file. A cancelled call,
by the idle rule or an interrupt, shuts its socket so the worker thread and the server's slot are
freed. A refused connection, a broken response or a status of 500 or above raises a retryable
`GenerationError`, as the API's connection errors do (`bko.generate.api`).

`bko.generate.weights.LOCAL_MODELS` names each model and its weights. The command fetches and
verifies them, prints the server's argv, and probes a running server with one call:

    python -m bko.generate.local fetch <model> [--data-dir <dir>]
    python -m bko.generate.local serve-command <model> [--data-dir <dir>]
    python -m bko.generate.local probe <model>

ADR `research/local-generator-backend` records the backend and why.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import http.client
import json
import logging
import shlex
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

from bko.datasets.errors import DatasetError
from bko.datasets.fetch import default_data_dir
from bko.datasets.uci import UCI_FRAME
from bko.generate.decode import decode, undecoded_in
from bko.generate.errors import GenerationError
from bko.generate.guidance import Guidance
from bko.generate.messages import SETTINGS, AsyncSend, Call, Reply, Usage, record_schema
from bko.generate.prompt import render
from bko.generate.weights import LOCAL_MODELS, fetch, model_path


logger = logging.getLogger(__name__)


LOCAL_URL: Final = "http://127.0.0.1:8081"
"""The address `llama-server` listens on; it binds the loopback interface only."""

SLOTS: Final = 4
"""Parallel slots the server keeps; a local run keeps at most this many calls in flight
(`bko.evaluate.scoring`)."""

CONTEXT: Final = 81920
"""The server's context in tokens, shared evenly by its slots."""

TIMEOUT_SECONDS: Final = 1800.0
"""The most one socket operation may wait, a whole generation included."""

READY_SECONDS: Final = 600.0
"""How long `probe` waits for a starting server's `/health` to answer, its weights loading."""

ERROR_TEXT: Final = 300
"""The most characters of a refusing server's body an error message quotes."""

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


def _count(counts: Mapping[str, object], key: str) -> int:
    value = counts.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def reply_of(body: object) -> Reply:
    """A chat-completions response as a `Reply`: its text, tokens and stop reason.

    A local model bills nothing, so no input is cached; the prompt's tokens are input tokens.

    Raises:
        GenerationError: Not retryable, if `body` is not a JSON object.
    """
    if not isinstance(body, dict):
        msg = f"the local server's response is not a JSON object: {type(body).__name__}"
        raise GenerationError(msg)
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    first = first if isinstance(first, dict) else {}
    message = first.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    finish = first.get("finish_reason")
    usage = body.get("usage")
    counts = usage if isinstance(usage, dict) else {}
    request_id = body.get("id")
    return Reply(
        request_id=request_id if isinstance(request_id, str) else None,
        model=str(body.get("model", "")),
        stop_reason=STOP_REASONS.get(finish, finish) if isinstance(finish, str) else None,
        text=text if isinstance(text, str) else None,
        usage=Usage(
            input_tokens=_count(counts, "prompt_tokens"),
            output_tokens=_count(counts, "completion_tokens"),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        raw=dict(body),
    )


def loopback(url: str) -> int:
    """`url`'s port, once `url` is shown to name the loopback server.

    Raises:
        ValueError: If `url` is not `http://127.0.0.1:<port>` with no user, path, query or fragment.
    """
    parts = urlsplit(url)
    bare = not (parts.username or parts.password or parts.path.strip("/") or parts.query)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or not bare or parts.fragment:
        msg = f"refusing {url}: a local model is served on the loopback interface only"
        raise ValueError(msg)
    if parts.port is None:
        msg = f"refusing {url}: name the server's port"
        raise ValueError(msg)
    return parts.port


def _exchange(
    connection: http.client.HTTPConnection, method: str, path: str, body: bytes | None = None
) -> object:
    """One request on `connection`, its JSON answer returned.

    Raises:
        GenerationError: Retryable for a failed connection, a broken response or a status of 500
            or above; not retryable for another refusing status or a body that is not JSON.
    """
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
    except (OSError, http.client.HTTPException) as exc:
        msg = f"the local server could not be reached or read: {type(exc).__name__}: {exc}"
        raise GenerationError(msg, retryable=True) from None
    if response.status >= 300:
        text = data[:ERROR_TEXT].decode("utf-8", "replace")
        msg = f"the local server refused the request: status={response.status} body={text!r}"
        raise GenerationError(msg, retryable=response.status >= RETRYABLE_STATUS)
    try:
        return json.loads(data)
    except ValueError:
        msg = f"the local server's response is not JSON: {data[:ERROR_TEXT]!r}"
        raise GenerationError(msg) from None


def check_server(model: str, port: int) -> Mapping[str, object]:
    """The server's `/props`, once its alias and weights file are shown to be `model`'s.

    Logs one `local server` line: the alias, the weights path, the build, the slots and the
    pinned sha256 of the file the path names.

    Raises:
        GenerationError: Not retryable, if the server serves another alias or file; as
            `_exchange` if it cannot be read.
    """
    pin = LOCAL_MODELS[model]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT_SECONDS)
    try:
        props = _exchange(connection, "GET", "/props")
    finally:
        connection.close()
    props = props if isinstance(props, dict) else {}
    alias, path = props.get("model_alias"), str(props.get("model_path", ""))
    if alias != model or Path(path).name != pin.filename:
        msg = f"the local server serves alias {alias} from {path}; the run names {model}"
        raise GenerationError(msg)
    logger.info(
        "local server model=%s path=%s build=%s slots=%s sha256=%s",
        model,
        path,
        props.get("build_info"),
        props.get("total_slots"),
        pin.sha256,
    )
    return props


def local_send(model: str, *, url: str = LOCAL_URL) -> AsyncSend:
    """Post each call to the llama.cpp server at `url`, in a worker thread.

    The first call checks the server (`check_server`). A call cancelled while it waits shuts its
    socket, so its thread returns and the server stops generating for it.

    Args:
        model: The model id the server serves under its alias; each call's settings name it.
        url: The server's address, `http://127.0.0.1:<port>`.

    Returns:
        A send whose replies come from the server.

    Raises:
        ValueError: If `url` is not a loopback address (`loopback`).
    """
    port = loopback(url)
    checked = threading.Event()
    lock = threading.Lock()

    def post(call: Call, connection: http.client.HTTPConnection) -> Reply:
        if call.settings.model != model:
            msg = f"the call names model {call.settings.model}; this server serves {model}"
            raise GenerationError(msg)
        with lock:
            if not checked.is_set():
                check_server(model, port)
                checked.set()
        data = json.dumps(payload(call)).encode()
        return reply_of(_exchange(connection, "POST", "/v1/chat/completions", data))

    async def send(call: Call) -> Reply:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT_SECONDS)
        try:
            return await asyncio.to_thread(post, call, connection)
        except asyncio.CancelledError:
            _abort(connection)
            raise
        finally:
            connection.close()

    return send


def _abort(connection: http.client.HTTPConnection) -> None:
    """Shut `connection`'s socket, so a read blocked on it returns at once."""
    if connection.sock is not None:
        try:
            connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            return


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
        str(loopback(LOCAL_URL)),
        "-np",
        str(SLOTS),
        "-c",
        str(CONTEXT),
        *rendering,
    ]


PROBE_CELL: Final = ("<55", "female", 1)
"""The cell the probe asks for: E1's smallest. Its guidance bounds each measured field by UCI's
plausible range, age below the band edge."""


def wait_ready(
    port: int, *, seconds: float = READY_SECONDS, sleep: Callable[[float], None] = time.sleep
) -> None:
    """Wait until the server on `port` answers `/health` with 200, its weights loaded.

    Raises:
        GenerationError: Retryable, if it does not within `seconds`.
    """
    deadline = time.monotonic() + seconds
    while True:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        try:
            connection.request("GET", "/health")
            if connection.getresponse().status == 200:
                return
        except OSError, http.client.HTTPException:
            pass
        finally:
            connection.close()
        if time.monotonic() > deadline:
            msg = f"the local server on port {port} was not ready within {seconds:g} s"
            raise GenerationError(msg, retryable=True)
        sleep(1.0)


def probe(model: str, *, url: str = LOCAL_URL, rows: int = 4) -> int:
    """Send one call for `rows` records of `PROBE_CELL` and log what decodes and how fast.

    Returns:
        The records that decoded, of `rows`.

    Raises:
        GenerationError: If the server cannot be reached, serves another model or refuses the call.
    """
    settings = dataclasses.replace(SETTINGS, model=model)
    interval = {r.field: (r.low, r.high) for r in UCI_FRAME.ranges}
    interval["age"] = (interval["age"][0], float(UCI_FRAME.edges[0] - 1))
    guidance = Guidance(cell=PROBE_CELL, sex=0, label=1, records=2, interval=interval)
    system, user = render(guidance, rows, settings.guidance, frame=UCI_FRAME)
    call = Call(settings, 0, 1, system, user, frame=UCI_FRAME)
    started = time.monotonic()
    reply = asyncio.run(local_send(model, url=url)(call))
    elapsed = time.monotonic() - started
    candidates = decode(reply, rows)
    decoded = len(candidates) - undecoded_in(candidates)
    logger.info(
        "probe model=%s asked=%d decoded=%d stop_reason=%s input_tokens=%d output_tokens=%d "
        "elapsed_s=%.1f request_id=%s",
        model,
        rows,
        decoded,
        reply.stop_reason,
        reply.usage.input_tokens,
        reply.usage.output_tokens,
        elapsed,
        reply.request_id,
    )
    if decoded < rows:
        logger.warning("probe model=%s decoded %d of %d records", model, decoded, rows)
    for candidate in candidates:
        logger.info("probe record %s", json.dumps(candidate, default=str, sort_keys=True))
    return decoded


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return its exit status: 0 done, 2 a fetch or probe failed.

    `probe` first waits up to `READY_SECONDS` for the server to load its weights.
    """
    parser = argparse.ArgumentParser(
        prog="python -m bko.generate.local",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("fetch", "serve-command", "probe"))
    parser.add_argument("model", choices=sorted(LOCAL_MODELS))
    parser.add_argument("--data-dir", type=Path, default=None, help="for fetch and serve-command")
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.command == "probe":
        return _probe(args.model)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        if args.command == "fetch":
            fetch(args.model, data_dir)
        else:
            sys.stdout.write(shlex.join(serve_command(args.model, data_dir)) + "\n")
    except (GenerationError, DatasetError) as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    return 0


def _probe(model: str) -> int:
    """`probe` once the server is ready: 0 when a record decodes, else 2."""
    try:
        wait_ready(loopback(LOCAL_URL))
        decoded = probe(model)
    except GenerationError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    if decoded == 0:
        logger.error("probe model=%s decoded no record", model)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
