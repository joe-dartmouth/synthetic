"""`python -m bko.slm` trains and scores through its `Context` seams, on a tiny model."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from bko.datasets.errors import ContractError
from bko.evaluate.errors import IdentityError
from bko.slm.__main__ import META_FILE, Context, Holdout, Request, adapter_dir, main
from bko.slm.train import ADAPTER_FILE, LORA
from bko.slm.weights import OLMO_3_7B
from tests._datasets_fixtures import FakeOpener
from tests._slm_fixtures import FRAME, frame_tokenizer, model, records


TRAIN = records(12)
TEST = records(7, seed=1)
WORDS = frame_tokenizer(pd.concat([TRAIN, TEST]))
FOLDER = Path("runs/e6-99")
TINY = dataclasses.replace(LORA, rank=4, alpha=8, learning_rate=1e-2)
STORE = "ab" * 32


def _holdout(request: Request, data_dir: Path) -> Holdout:
    """`none`'s training set is `TRAIN`'s first 8 records, every other arm's all 12."""
    del data_dir
    assert request.folder == FOLDER
    train = TRAIN.iloc[:8] if request.arm == "none" else TRAIN
    return Holdout(FRAME, train, TEST, (), STORE)


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
    assert json.loads((path / META_FILE).read_text(encoding="utf-8")) == {
        "arm": "smote",
        "base_repo": OLMO_3_7B.repo,
        "base_revision": OLMO_3_7B.revision,
        "folder": "e6-99",
        "holdout": 0,
        "rows_sha256": STORE,
        "run": "e6-thina",
    }
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


def test_score_reads_the_adapter_of_its_arm(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    scored: dict[str, str] = {}
    for arm in ("none", "smote"):
        assert _run(tmp_path, "train", *_holdout_args("--arm", arm))[0] == 0
    for arm in ("none", "smote"):
        caplog.clear()
        code, scored[arm] = _run(tmp_path, "score", *_holdout_args("--arm", arm))
        assert code == 0
        assert f"path={adapter_dir(tmp_path, FOLDER, 0, arm)} sha256=" in caplog.text
    assert scored["none"] != scored["smote"]


@pytest.mark.parametrize("stray", ["adapter_model.bin", "README.md", "notes.txt"])
def test_score_refuses_an_adapter_directory_holding_another_file(tmp_path: Path, stray) -> None:
    assert _run(tmp_path, "train", *_holdout_args("--arm", "smote"))[0] == 0
    (adapter_dir(tmp_path, FOLDER, 0, "smote") / stray).write_bytes(b"x")
    assert _run(tmp_path, "score", *_holdout_args("--arm", "smote")) == (3, "")


@pytest.mark.parametrize(
    ("key", "value"),
    [("arm", "none"), ("run", "e6"), ("rows_sha256", "cd" * 32), ("base_revision", "0" * 40)],
)
def test_score_refuses_an_adapter_whose_meta_differs(tmp_path: Path, caplog, key, value) -> None:
    assert _run(tmp_path, "train", *_holdout_args("--arm", "smote"))[0] == 0
    path = adapter_dir(tmp_path, FOLDER, 0, "smote") / META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, key: value}), encoding="utf-8")
    assert _run(tmp_path, "score", *_holdout_args("--arm", "smote")) == (3, "")
    assert f"differs from this score's arguments at ['{key}']" in caplog.text


def test_score_refuses_an_adapter_without_its_meta(tmp_path: Path) -> None:
    assert _run(tmp_path, "train", *_holdout_args("--arm", "smote"))[0] == 0
    (adapter_dir(tmp_path, FOLDER, 0, "smote") / META_FILE).unlink()
    assert _run(tmp_path, "score", *_holdout_args("--arm", "smote")) == (3, "")


@pytest.mark.parametrize(
    ("error", "code"),
    [(OSError("unreadable"), 2), (ContractError("bad record"), 3), (IdentityError("moved"), 3)],
)
@pytest.mark.parametrize("command", ["train", "score"])
def test_a_failed_read_exits_with_its_code(tmp_path: Path, error, code, command) -> None:
    def failing(request: Request, data_dir: Path) -> Holdout:
        raise error

    broken = dataclasses.replace(CONTEXT, holdout=failing)
    argv = ["--data-dir", str(tmp_path), command, *_holdout_args("--arm", "none")]
    assert main(argv, context=broken, out=io.StringIO()) == code
