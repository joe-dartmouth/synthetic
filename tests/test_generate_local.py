"""`bko.generate.local`: the llama.cpp send, its server check, the server argv and the probe."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import bko.generate.local as local_module
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.scoring import _in_flight
from bko.generate.calls import ask
from bko.generate.decode import decode, undecoded_in
from bko.generate.errors import GenerationError
from bko.generate.local import (
    PREFLIGHT_TOKENS,
    SLOTS,
    local_send,
    loopback,
    main,
    payload,
    probe,
    probe_call,
    reply_of,
    serve_command,
    wait_ready,
)
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply, record_schema
from bko.generate.reasoning import reasoning_chars
from bko.generate.replay import Replay
from bko.generate.weights import LOCAL_MODELS, model_path
from tests._generate_fixtures import GUIDANCE


MODEL = "olmo-3-7b-instruct-q8_0"
BASE = "gemma-4-26b-a4b-q8_0"
THINKING = "gemma-4-26b-a4b-q8_0-thinking"
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
PROPS: dict[str, object] = {
    "model_alias": MODEL,
    "model_path": f"/w/{LOCAL_MODELS[MODEL].filename}",
    "build_info": "b1-test",
    "total_slots": 4,
}


def completion(
    records: list[dict[str, int | float]],
    finish: str = "stop",
    reasoning_content: object = None,
    content: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": json.dumps({"records": records}) if content is None else content,
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-1",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": message,
            }
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 120},
    }


class Server:
    """A llama-server fake on an ephemeral loopback port: `/props`, `/health`, each chat POST
    answered in turn with a status and a body (bytes sent as they are, else JSON), and each
    `/tokenize` POST from its own answers, three tokens once they run out.

    Attributes:
        requests: Each chat POST's method, path, content type and JSON body.
        tokenized: Each `/tokenize` POST's JSON body.
        closed: Set when a delayed POST sees the client hang up.
    """

    def __init__(
        self,
        answers: list[tuple[int, object]],
        *,
        props: dict[str, object] = PROPS,
        delay: float = 0.0,
        tokenize: list[tuple[int, object]] | None = None,
    ) -> None:
        self.answers = answers
        self.tokenize = tokenize if tokenize is not None else []
        self.requests: list[tuple[str, str, str, object]] = []
        self.tokenized: list[object] = []
        self.closed = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: object) -> None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._send(200, props if self.path == "/props" else {"status": "ok"})

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                if self.path == "/tokenize":
                    fake.tokenized.append(body)
                    answer = fake.tokenize.pop(0) if fake.tokenize else (200, {"tokens": [0] * 3})
                    self._send(*answer)
                    return
                kind = self.headers["Content-Type"]
                fake.requests.append((self.command, self.path, kind, body))
                if delay:
                    # Wait for the client to hang up, as llama-server notices a closed socket.
                    self.connection.settimeout(delay)
                    try:
                        if self.connection.recv(1) == b"":
                            fake.closed.set()
                            return
                    except TimeoutError:
                        pass
                self._send(*fake.answers.pop(0))

            def log_message(self, format: str, *args: object) -> None:  # ruff: ignore[builtin-argument-shadowing]
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
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


def test_a_rewrite_calls_payload_carries_its_own_schema():
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    call = dataclasses.replace(CALL, schema=schema)
    assert json.dumps(schema) in json.dumps(payload(call)["response_format"])
    assert json.dumps(schema) in json.dumps(call.params()["output_config"])
    assert call.reply_schema() == schema


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


def test_a_malformed_response_is_read_safely_or_refused():
    reply = reply_of({"usage": {"prompt_tokens": None, "completion_tokens": True}, "choices": []})
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (0, 0)
    assert (reply.text, reply.stop_reason) == (None, None)
    with pytest.raises(GenerationError, match="not a JSON object"):
        reply_of([1, 2])


def test_the_send_checks_the_server_once_and_posts_the_payload(caplog):
    caplog.set_level("INFO")
    with Server([(200, completion([RECORD])), (200, completion([RECORD]))]) as fake:
        send = local_send(MODEL, url=fake.url)
        replies = [asyncio.run(send(CALL)) for _ in range(2)]
    request = ("POST", "/v1/chat/completions", "application/json", payload(CALL))
    assert fake.requests == [request, request]
    assert fake.answers == []
    assert all(reply.text == json.dumps({"records": [RECORD]}) for reply in replies)
    lines = [m for m in caplog.messages if m.startswith("local server ")]
    assert lines == [
        (
            f"local server model={MODEL} thinking=false path={PROPS['model_path']} "
            f"build=b1-test slots=4 sha256={LOCAL_MODELS[MODEL].sha256}"
        )
    ]


@pytest.mark.parametrize(
    "props",
    [{**PROPS, "model_path": "/w/other.gguf"}, {**PROPS, "model_alias": "gemma-4-26b-a4b-q8_0"}],
    ids=["file", "alias"],
)
def test_a_server_serving_another_model_is_refused_before_any_post(props):
    with (
        Server([(200, completion([RECORD]))], props=props) as fake,
        pytest.raises(GenerationError, match="the run names") as caught,
    ):
        asyncio.run(local_send(MODEL, url=fake.url)(CALL))
    assert not caught.value.retryable
    assert fake.requests == []


def test_a_refused_connection_raises_the_retryable_error():
    with Server([]) as fake:
        url = fake.url
    with pytest.raises(GenerationError, match="could not be reached") as caught:
        asyncio.run(local_send(MODEL, url=url)(CALL))
    assert caught.value.retryable


def test_server_errors_and_broken_bodies_raise_with_the_right_retry():
    answers: list[tuple[int, object]] = [
        (500, {"error": "Loading model"}),
        (400, {"error": "context size exceeded"}),
        (200, b"not json"),
    ]
    with Server(answers) as fake:
        send = local_send(MODEL, url=fake.url)
        with pytest.raises(GenerationError, match=r"status=500.*Loading model") as first:
            asyncio.run(send(CALL))
        with pytest.raises(GenerationError, match="status=400") as second:
            asyncio.run(send(CALL))
        with pytest.raises(GenerationError, match="not JSON") as third:
            asyncio.run(send(CALL))
    assert first.value.retryable
    assert not second.value.retryable
    assert not third.value.retryable


def test_a_socket_timeout_is_a_retryable_error(monkeypatch):
    monkeypatch.setattr(local_module, "TIMEOUT_SECONDS", 0.3)
    with (
        Server([(200, completion([RECORD]))], delay=3.0) as fake,
        pytest.raises(GenerationError, match="TimeoutError") as caught,
    ):
        asyncio.run(local_send(MODEL, url=fake.url)(CALL))
    assert caught.value.retryable


def test_the_send_ignores_proxy_settings(server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    reply = asyncio.run(local_send(MODEL, url=server.url)(CALL))
    assert reply.text == json.dumps({"records": [RECORD]})


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org:8081",
        "http://127.0.0.1:80@evil.example:9",
        "https://127.0.0.1:8081",
        "http://127.0.0.1",
        "http://127.0.0.1:8081/v1",
        "http://user@127.0.0.1:8081",
    ],
)
def test_only_a_bare_loopback_address_is_accepted(url):
    with pytest.raises(ValueError, match="refusing"):
        loopback(url)
    with pytest.raises(ValueError, match="refusing"):
        local_send(MODEL, url=url)


def test_the_send_refuses_another_model(server):
    other = dataclasses.replace(CALL, settings=SETTINGS)
    with pytest.raises(GenerationError, match="serves"):
        asyncio.run(local_send(MODEL, url=server.url)(other))
    assert server.requests == []


def test_a_cancelled_call_hangs_up_and_frees_its_thread():
    """The idle rule cancels the await; the send shuts its socket, so the server sees the hang-up
    and the worker thread returns well before the server would have answered."""
    with Server([(200, completion([RECORD]))], delay=5.0) as fake:
        send = local_send(MODEL, url=fake.url)

        async def cut() -> None:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.3):
                    await send(CALL)

        started = time.monotonic()
        asyncio.run(cut())
        assert fake.closed.wait(2.0)
    assert time.monotonic() - started < 3.0


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
    assert serve_command(MODEL, tmp_path)[-2:] == ["--chat-template", "chatml"]


def test_probe_decodes_one_valid_call(server, caplog):
    caplog.set_level("INFO")
    assert probe(MODEL, url=server.url, rows=1) == 1
    assert any(m.startswith(f"probe model={MODEL} asked=1 decoded=1") for m in caplog.messages)


def test_wait_ready_returns_once_health_answers_and_gives_up_after_its_limit(server):
    wait_ready(server.port, seconds=1.0)
    with Server([]) as fake:
        port = fake.port
    with pytest.raises(GenerationError, match="not ready") as caught:
        wait_ready(port, seconds=0.0, sleep=lambda seconds: None)
    assert caught.value.retryable


def test_the_command_prints_the_serve_argv(tmp_path, capsys):
    assert main(["serve-command", MODEL, "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"llama-server -m {model_path(MODEL, tmp_path)} --alias {MODEL} ")
    assert out.endswith("--chat-template chatml\n")


@pytest.mark.parametrize(("decoded", "code"), [(0, 2), (1, 0)])
def test_a_probe_exits_two_when_nothing_decodes(decoded, code, monkeypatch):
    monkeypatch.setattr(local_module, "wait_ready", lambda port: None)
    monkeypatch.setattr(local_module, "probe", lambda model: decoded)
    assert main(["probe", MODEL]) == code


def test_a_server_that_never_readies_exits_two(monkeypatch):
    def refuse(port: int) -> None:
        raise GenerationError("unreachable")

    monkeypatch.setattr(local_module, "wait_ready", refuse)
    assert main(["probe", MODEL]) == 2


def test_a_local_run_keeps_at_most_the_servers_slots_in_flight(caplog):
    caplog.set_level("INFO")
    assert _in_flight(8, MODEL) == SLOTS
    assert "local in_flight=4 slots=4 asked=8" in caplog.messages
    assert _in_flight(2, MODEL) == 2
    assert _in_flight(8, SETTINGS.model) == 8


def test_a_local_log_replays_its_text_and_decodes_as_logged(tmp_path):
    settings = dataclasses.replace(SETTINGS, model=MODEL)
    sent: list[Call] = []
    body = completion([RECORD, RECORD])

    def send(call: Call) -> Reply:
        sent.append(call)
        return reply_of(body)

    path = tmp_path / "calls.jsonl"
    with CallLog(path, settings, frame="uci") as log:
        batch = ask(GUIDANCE, 2, send=send, log=log, frame=UCI_FRAME)
    logged = (log.decoded, log.undecoded)
    assert logged == (2, 0)
    replayed = Replay(path)(sent[0])
    assert replayed.text is not None
    assert replayed.text == json.dumps({"records": [RECORD, RECORD]})
    with CallLog(tmp_path / "again.jsonl", settings, frame="uci") as again:
        ask(GUIDANCE, 2, send=Replay(path), log=again, frame=UCI_FRAME)
    assert (again.decoded, again.undecoded) == logged
    assert len(batch.candidates) == 2


THINKING_CALL = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=THINKING))
THINKING_PROPS = {
    **PROPS,
    "model_alias": THINKING,
    "model_path": f"/w/{LOCAL_MODELS[BASE].filename}",
}
PREFLIGHT = (200, completion([], finish="length", reasoning_content="plan", content=""))


def test_the_thinking_payload_differs_from_the_plain_one_in_model_and_thinking_alone():
    plain = payload(dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=BASE)))
    thinking = payload(THINKING_CALL)
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}
    assert thinking["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: v for k, v in thinking.items() if k not in {"model", "chat_template_kwargs"}} == {
        k: v for k, v in plain.items() if k not in {"model", "chat_template_kwargs"}
    }
    assert (plain["model"], thinking["model"]) == (BASE, THINKING)


def test_the_thinking_argv_differs_in_alias_and_reasoning_flags_alone(tmp_path):
    plain = serve_command(BASE, tmp_path)
    thinking = serve_command(THINKING, tmp_path)
    flags = ["--jinja", "--reasoning", "on", "--reasoning-format", "deepseek"]
    assert thinking[-5:] == flags
    assert plain[-3:] == ["--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']
    assert [BASE if a == THINKING else a for a in thinking[:-5]] == plain[:-3]
    assert thinking == [
        "llama-server",
        "-m",
        str(model_path(BASE, tmp_path)),
        "--alias",
        THINKING,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "-np",
        "4",
        "-c",
        "81920",
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    ]


def test_a_base_alias_server_is_refused_for_the_variant_before_any_post():
    props = {**PROPS, "model_alias": BASE, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    with (
        Server([(200, completion([RECORD]))], props=props) as fake,
        pytest.raises(GenerationError, match="the run names") as caught,
    ):
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    assert not caught.value.retryable
    assert fake.requests == []


def test_the_probe_logs_reasoning_chars(caplog):
    caplog.set_level("INFO")
    body = completion([RECORD], reasoning_content="think hard")
    with Server([(200, body), (200, completion([RECORD]))]) as fake:
        assert probe(MODEL, url=fake.url, rows=1) == 1
        assert probe(MODEL, url=fake.url, rows=1) == 1
    lines = [m for m in caplog.messages if m.startswith("probe model=")]
    assert "reasoning_chars=10" in lines[0]
    assert "reasoning_chars=0" in lines[1]


def test_a_probe_whose_content_holds_think_tags_decodes_nothing_and_exits_two(monkeypatch):
    tagged = "<think>plan</think>" + json.dumps({"records": [RECORD]})
    with Server([(200, completion([RECORD], content=tagged))]) as fake:
        monkeypatch.setattr(local_module, "wait_ready", lambda port: None)
        original = probe
        monkeypatch.setattr(local_module, "probe", lambda model: original(model, url=fake.url))
        assert main(["probe", MODEL]) == 2


def test_the_server_line_names_thinking_for_a_variant(caplog):
    caplog.set_level("INFO")
    with Server([PREFLIGHT, (200, completion([RECORD]))], props=THINKING_PROPS) as fake:
        asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    lines = [m for m in caplog.messages if m.startswith("local server ")]
    assert [m.split()[:4] for m in lines] == [
        ["local", "server", f"model={THINKING}", "thinking=true"]
    ]


def test_a_thinking_server_alias_is_refused_for_a_base_run():
    props = {**PROPS, "model_alias": THINKING, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    base_call = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=BASE))
    with (
        Server([(200, completion([RECORD]))], props=props) as fake,
        pytest.raises(GenerationError, match="the run names") as caught,
    ):
        asyncio.run(local_send(BASE, url=fake.url)(base_call))
    assert not caught.value.retryable
    assert fake.requests == []


def test_an_unlisted_model_has_no_payload():
    call = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model="no-such-model"))
    with pytest.raises(GenerationError, match="no-such-model") as caught:
        payload(call)
    assert not caught.value.retryable


def test_a_thinking_reply_cut_at_the_budget_decodes_nothing_and_keeps_its_tokens():
    body = completion([RECORD], finish="length", reasoning_content="t" * 50, content="")
    reply = reply_of(body)
    assert (reply.stop_reason, reply.text) == ("max_tokens", "")
    assert reply.usage.output_tokens == 120
    assert reasoning_chars(reply) == 50
    candidates = decode(reply, 1)
    assert len(candidates) - undecoded_in(candidates) == 0


def test_a_thinking_reply_decodes_from_content_alone():
    reply = reply_of(completion([RECORD], reasoning_content="not json {"))
    candidates = decode(reply, 1)
    assert candidates[: len(candidates) - undecoded_in(candidates)] == (RECORD,)


def test_a_thinking_probe_with_no_reasoning_warns_and_exits_two(monkeypatch, caplog):
    with Server([PREFLIGHT, (200, completion([RECORD]))], props=THINKING_PROPS) as fake:
        monkeypatch.setattr(local_module, "wait_ready", lambda port: None)
        original = probe
        monkeypatch.setattr(local_module, "probe", lambda model: original(model, url=fake.url))
        with caplog.at_level("INFO"):
            assert main(["probe", THINKING]) == 2
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("reasoning_chars=0" in m and THINKING in m for m in warnings)


def test_a_thinking_probe_with_reasoning_exits_zero(monkeypatch):
    props = {**PROPS, "model_alias": THINKING, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    body = completion([RECORD], reasoning_content="think")
    with Server([PREFLIGHT, (200, body)], props=props) as fake:
        monkeypatch.setattr(local_module, "wait_ready", lambda port: None)
        original = probe
        monkeypatch.setattr(local_module, "probe", lambda model: original(model, url=fake.url))
        assert main(["probe", THINKING]) == 0


def test_a_thinking_id_whose_preflight_holds_no_reasoning_is_refused(monkeypatch, caplog):
    """A server that does not parse reasoning out, as under `--reasoning-format none`, answers
    the pre-flight with its thoughts in the content; the run stops before its first call."""
    unsplit = completion([], finish="length", content="<|channel>thought\n plan")
    with Server([(200, unsplit), (200, completion([RECORD]))], props=THINKING_PROPS) as fake:
        with pytest.raises(GenerationError, match="does not reason") as caught:
            asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
        monkeypatch.setattr(local_module, "wait_ready", lambda port: None)
        original = probe
        monkeypatch.setattr(local_module, "probe", lambda model: original(model, url=fake.url))
        fake.answers.insert(0, (200, unsplit))
        with caplog.at_level("INFO"):
            assert main(["probe", THINKING]) == 2
    assert not caught.value.retryable
    preflight = payload(probe_call(THINKING, 4, max_tokens=PREFLIGHT_TOKENS))
    assert [request[3] for request in fake.requests] == [preflight, preflight]
    assert preflight["max_tokens"] == PREFLIGHT_TOKENS == 64
    assert fake.answers == [(200, completion([RECORD]))]
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    refusal = f"the local server for {THINKING} does not reason: its pre-flight reply has"
    assert errors == [f"{refusal} reasoning_chars=0"]
    assert not [m for m in caplog.messages if m.startswith("local server ")]


def test_a_thinking_id_whose_preflight_reasons_passes(caplog):
    caplog.set_level("INFO")
    with Server([PREFLIGHT, (200, completion([RECORD]))], props=THINKING_PROPS) as fake:
        reply = asyncio.run(local_send(THINKING, url=fake.url)(THINKING_CALL))
    assert reply.text == json.dumps({"records": [RECORD]})
    preflight = payload(probe_call(THINKING, 4, max_tokens=PREFLIGHT_TOKENS))
    assert [request[3] for request in fake.requests] == [preflight, payload(THINKING_CALL)]
    lines = [m for m in caplog.messages if m.startswith("local server ")]
    assert lines == [
        (
            f"local server model={THINKING} thinking=true path={THINKING_PROPS['model_path']} "
            f"build=b1-test slots=4 sha256={LOCAL_MODELS[THINKING].sha256} "
            "preflight_reasoning_chars=4"
        )
    ]


def test_a_base_id_sends_no_preflight():
    props = {**PROPS, "model_alias": BASE, "model_path": f"/w/{LOCAL_MODELS[BASE].filename}"}
    base_call = dataclasses.replace(CALL, settings=dataclasses.replace(SETTINGS, model=BASE))
    with Server([(200, completion([RECORD]))], props=props) as fake:
        asyncio.run(local_send(BASE, url=fake.url)(base_call))
    assert [request[3] for request in fake.requests] == [payload(base_call)]
    assert fake.tokenized == []
