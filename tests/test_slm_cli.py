"""`python -m bko.slm` trains and scores through its `Context` seams, on a tiny model."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from bko.slm.__main__ import Context, Holdout, Request, adapter_dir, main
from bko.slm.train import ADAPTER_FILE, LORA
from tests._datasets_fixtures import FakeOpener
from tests._slm_fixtures import FRAME, frame_tokenizer, model, records


TRAIN = records(12)
TEST = records(7, seed=1)
WORDS = frame_tokenizer(pd.concat([TRAIN, TEST]))
FOLDER = Path("runs/e6-99")
TINY = dataclasses.replace(LORA, rank=4, alpha=8, learning_rate=1e-2)


def _holdout(request: Request, data_dir: Path) -> Holdout:
    del data_dir
    assert request.folder == FOLDER
    return Holdout(FRAME, TRAIN, TEST, ())


def _load(*_: object, **__: object) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    return model(len(WORDS)), WORDS


def _no_network(url: str, timeout: float) -> BinaryIO:
    msg = f"the command reached the network: {url} {timeout}"
    raise AssertionError(msg)


CONTEXT = Context(
    open_url=_no_network,
    load_base=_load,
    device=lambda: torch.device("cpu"),
    holdout=_holdout,
    settings=TINY,
)


def _run(tmp_path: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    common = ["--data-dir", str(tmp_path)]
    code = main([*common, *argv], context=CONTEXT, out=out)
    return code, out.getvalue()


def _holdout_args(*extra: str) -> list[str]:
    return ["e6-thina", "--folder", str(FOLDER), "--rows", "rows.jsonl", "--holdout", "0", *extra]


def test_train_writes_an_adapter_and_logs_its_sha256(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    code, text = _run(tmp_path, "train", *_holdout_args("--arm", "smote"))
    assert (code, text) == (0, "")
    path = adapter_dir(tmp_path, FOLDER, 0, "smote")
    assert path == tmp_path / "models" / "slm-adapters" / "e6-99" / "0-smote"
    weights = path / ADAPTER_FILE
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    size = weights.stat().st_size
    assert f"adapter bytes={size} path={path} sha256={digest}" in caplog.text
    assert "train holdout=0 arm=smote rows=12" in caplog.text
    assert re.search(r"step=3 of=3 loss=\S+ lr=\S+ seconds=\S+", caplog.text)
    assert re.search(r"train seconds=\S+ seconds_per_row=\S+ final_loss=\S+", caplog.text)


def test_score_writes_one_probability_per_test_record(tmp_path: Path, caplog) -> None:
    assert _run(tmp_path, "train", *_holdout_args("--arm", "none"))[0] == 0
    caplog.set_level(logging.INFO)
    caplog.clear()
    code, text = _run(tmp_path, "score", *_holdout_args("--arm", "none"))
    assert code == 0
    scored = pd.read_csv(io.StringIO(text), index_col="record_id")
    assert list(scored.columns) == ["probability"]
    assert scored.index.tolist() == TEST.index.tolist()
    assert ((scored["probability"] > 0) & (scored["probability"] < 1)).all()
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"probabilities rows={len(TEST)} sha256={sha}" in caplog.text
    assert "measure name=brier scope=all value=" in caplog.text
    assert "adapter bytes=" in caplog.text
    code, base = _run(tmp_path, "score", *_holdout_args("--base"))
    assert code == 0
    assert base != text


def test_score_refuses_an_adapter_that_is_absent(tmp_path: Path) -> None:
    assert _run(tmp_path, "score", *_holdout_args("--arm", "smote")) == (2, "")


def test_train_refuses_an_existing_adapter_path(tmp_path: Path) -> None:
    adapter_dir(tmp_path, FOLDER, 0, "smote").mkdir(parents=True)
    assert _run(tmp_path, "train", *_holdout_args("--arm", "smote")) == (2, "")


def test_without_mps_the_command_cannot_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    no_mps = dataclasses.replace(CONTEXT, device=Context().device)
    argv = ["--data-dir", str(tmp_path), "train", *_holdout_args("--arm", "none")]
    assert main(argv, context=no_mps, out=io.StringIO()) == 2


def test_a_failed_fetch_exits_2_and_keeps_no_part_file(tmp_path: Path) -> None:
    failing = dataclasses.replace(CONTEXT, open_url=FakeOpener(error=OSError("no route")))
    argv = ["--data-dir", str(tmp_path), "fetch"]
    assert main(argv, context=failing, out=io.StringIO()) == 2
    assert not list(tmp_path.rglob("*.part"))


def test_the_commands_make_no_api_call(tmp_path: Path) -> None:
    assert _run(tmp_path, "train", *_holdout_args("--arm", "real"))[0] == 0
    assert _run(tmp_path, "score", *_holdout_args("--arm", "real"))[0] == 0
