"""`bko.generate.local`: the llama.cpp send, the weights pins, the server argv and the probe."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import bko.generate.local as local_module
from bko.datasets.uci import UCI_FRAME
from bko.generate.errors import GenerationError
from bko.generate.local import (
    LOCAL_MODELS,
    LocalModel,
    fetch,
    local_send,
    main,
    model_path,
    payload,
    probe,
    reply_of,
    serve_command,
    verify_model,
)
from bko.generate.messages import SETTINGS, Call, record_schema


MODEL = "olmo-3-7b-instruct-q8_0"
CALL = Call(dataclasses.replace(SETTINGS, model=MODEL), 0, 1, "sys", "usr", frame=UCI_FRAME)
RECORD: dict[str, int | float] = {
    "age": 45,
    "sex": 0,
    "cp": 4,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 7,
    "target": 1,
}


def completion(records: list[dict[str, int | float]], finish: str = "stop") -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": json.dumps({"records": records})},
            }
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 120},
    }


class Server:
    """A chat-completions fake on an ephemeral loopback port, answering each POST in turn."""

    def __init__(self, answers: list[tuple[int, dict[str, object]]]) -> None:
        self.answers = answers
        self.bodies: list[dict[str, object]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                fake.bodies.append(json.loads(self.rfile.read(length)))
                status, body = fake.answers.pop(0)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:  # ruff: ignore[builtin-argument-shadowing]
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> Server:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[Server]:
    with Server([(200, completion([RECORD]))]) as fake:
        yield fake


def test_the_payload_carries_both_turns_the_schema_and_fixed_sampling():
    assert payload(CALL) == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        "max_tokens": SETTINGS.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "records",
                "strict": True,
                "schema": record_schema(0, 1, frame=UCI_FRAME),
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_a_response_becomes_a_reply():
    body = completion([RECORD])
    reply = reply_of(body)
    assert reply.text == json.dumps({"records": [RECORD]})
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (900, 120)
    assert (reply.usage.cache_creation_input_tokens, reply.usage.cache_read_input_tokens) == (0, 0)
    assert reply.stop_reason == "end_turn"
    assert reply.request_id == "chatcmpl-1"
    assert reply.model == MODEL
    assert reply.raw == body
    assert reply_of(completion([], finish="length")).stop_reason == "max_tokens"


def test_the_send_posts_the_payload_and_reads_the_reply(server):
    reply = asyncio.run(local_send(MODEL, url=server.url)(CALL))
    assert server.bodies == [payload(CALL)]
    assert reply.text == json.dumps({"records": [RECORD]})


def test_a_refused_connection_raises_the_retryable_error():
    with Server([]) as fake:
        url = fake.url
    with pytest.raises(GenerationError, match="could not be reached") as caught:
        asyncio.run(local_send(MODEL, url=url)(CALL))
    assert caught.value.retryable


def test_a_server_error_raises_retryable_and_a_client_error_not():
    with Server([(500, {"error": "x"}), (400, {"error": "y"})]) as fake:
        send = local_send(MODEL, url=fake.url)
        with pytest.raises(GenerationError, match="status=500") as first:
            asyncio.run(send(CALL))
        with pytest.raises(GenerationError, match="status=400") as second:
            asyncio.run(send(CALL))
    assert first.value.retryable
    assert not second.value.retryable


def test_the_send_refuses_another_model_and_a_non_loopback_address(server):
    other = dataclasses.replace(CALL, settings=SETTINGS)
    with pytest.raises(GenerationError, match="serves"):
        asyncio.run(local_send(MODEL, url=server.url)(other))
    with pytest.raises(ValueError, match="loopback"):
        local_send(MODEL, url="http://example.org:8081")


PIN_BYTES = b"weights" * 1000
PIN = LocalModel(
    repo="org/repo",
    filename="w.gguf",
    size=len(PIN_BYTES),
    sha256=hashlib.sha256(PIN_BYTES).hexdigest(),
)


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(local_module, "LOCAL_MODELS", {"toy": PIN})
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


def test_a_download_that_differs_from_the_pin_is_refused_and_not_published(pinned, tmp_path):
    with pytest.raises(GenerationError, match="the pin is"):
        fetch(pinned, tmp_path, open_url=lambda url, timeout: io.BytesIO(PIN_BYTES[:-1]))
    assert not model_path(pinned, tmp_path).exists()


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


def test_serve_command_pins_alias_slots_context_and_thinking(tmp_path):
    model = "gemma-4-26b-a4b-q8_0"
    assert serve_command(model, tmp_path) == [
        "llama-server",
        "-m",
        str(model_path(model, tmp_path)),
        "--alias",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "-np",
        "4",
        "-c",
        "81920",
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]


def test_a_model_with_a_built_in_template_is_served_with_it(tmp_path):
    assert serve_command(MODEL, tmp_path) == [
        "llama-server",
        "-m",
        str(model_path(MODEL, tmp_path)),
        "--alias",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "-np",
        "4",
        "-c",
        "81920",
        "--chat-template",
        "chatml",
    ]


def test_probe_decodes_one_valid_call(server, caplog):
    caplog.set_level("INFO")
    assert probe(MODEL, url=server.url, rows=1) == 1
    assert any(m.startswith(f"probe model={MODEL} asked=1 decoded=1") for m in caplog.messages)


def test_the_command_prints_the_serve_argv(tmp_path, capsys):
    assert main(["serve-command", MODEL, "--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("llama-server -m ")


def test_a_failed_probe_exits_two(monkeypatch):
    def refuse(model: str) -> int:
        raise GenerationError("unreachable")

    monkeypatch.setattr(local_module, "probe", refuse)
    assert main(["probe", MODEL]) == 2


def test_the_weights_root_is_under_the_data_dir(tmp_path):
    assert model_path(MODEL, tmp_path).is_relative_to(tmp_path / "models" / MODEL)
    assert isinstance(model_path(MODEL, tmp_path), Path)
