"""`Context.sender`: an on-device model's calls go to `local_send`, an API model's to `send`."""

from __future__ import annotations

import argparse
import dataclasses
import io

import pytest

import bko.evaluate.scoring as scoring_module
from bko.evaluate.__main__ import main
from bko.evaluate.named_runs import NamedRun
from bko.generate.messages import SETTINGS, Call
from bko.generate.weights import LOCAL_MODELS
from tests._evaluate_bko_fixtures import NARROW, context, score_argv, scripted


LOCAL = "gemma-4-26b-a4b-q8_0"


def refuse(call: Call):
    pytest.fail("an API send was built for a local model's run")


def test_the_sender_picks_local_send_by_model():
    built: list[str] = []
    ctx, _ = context(refuse)
    ctx = dataclasses.replace(ctx, local_send=lambda model: built.append(model) or refuse)
    assert set(LOCAL_MODELS) >= {LOCAL}
    ctx.sender(LOCAL)()
    assert built == [LOCAL]
    assert ctx.sender(SETTINGS.model) is ctx.send


def test_a_local_model_sends_through_the_local_factory(tmp_path):
    answers = scripted()
    built: list[str] = []

    def local(model: str):
        built.append(model)
        return answers

    ctx, _ = context(refuse)
    run = NamedRun(NARROW, dataclasses.replace(SETTINGS, model=LOCAL))
    ctx = dataclasses.replace(ctx, runs={"x1": run}, local_send=local, send=boom)
    argv = score_argv(tmp_path / "calls.jsonl", tmp_path, "x1")
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    assert built == [LOCAL]
    answers.assert_consumed()


def test_an_api_model_never_reaches_the_local_factory(tmp_path):
    answers = scripted()

    def local(model: str):
        pytest.fail("a local send was built for an API model's run")

    ctx, _ = context(answers)
    ctx = dataclasses.replace(ctx, local_send=local)
    assert main(score_argv(tmp_path / "calls.jsonl", tmp_path), context=ctx) == 0
    answers.assert_consumed()


def boom():
    pytest.fail("an API send was built for a local model's run")


def test_a_resume_of_a_local_run_sends_through_the_local_factory(monkeypatch, tmp_path):
    answers = scripted()
    captured: list[object] = []

    class Recorded:
        def __init__(self, path, send, *, ceiling) -> None:
            captured.append(send)
            raise SystemExit(0)

    monkeypatch.setattr(scoring_module, "Resume", Recorded)
    ctx, _ = context(answers)
    ctx = dataclasses.replace(ctx, send=boom, local_send=lambda model: answers)
    args = argparse.Namespace(replay=None, resume=tmp_path / "calls.jsonl")
    run = NamedRun(NARROW, dataclasses.replace(SETTINGS, model=LOCAL))
    with pytest.raises(SystemExit):
        scoring_module._answers(args, ctx, run)
    assert captured == [answers]
