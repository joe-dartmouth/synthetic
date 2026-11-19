"""`video.render`: the concat list it hands FFmpeg and the exit code of each failure."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from video import render
from video.render import concat_lines


pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture
def media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the driver's output paths into `tmp_path` and pretend both tools are installed."""
    scenes = tmp_path / "scenes"
    monkeypatch.setattr(render, "SCENES", scenes)
    monkeypatch.setattr(render, "OUTPUT", tmp_path / "bko-demo.mp4")
    monkeypatch.setattr(render, "PARTIAL", tmp_path / "bko-demo.part.mp4")
    monkeypatch.setattr(render.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render.importlib.util, "find_spec", lambda _: object())
    return scenes


type Step = int | Callable[[], object] | BaseException


def _runs(monkeypatch: pytest.MonkeyPatch, *steps: Step) -> None:
    """Replace `subprocess.run` with one step per call: a return code, a file writer, or a raise."""
    pending = list(steps)

    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        step = pending.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, int):
            return subprocess.CompletedProcess(argv, step)
        step()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(render.subprocess, "run", run)


def _write_scenes(scenes: Path) -> None:
    scenes.mkdir(parents=True, exist_ok=True)
    for name in render.ORDER:
        (scenes / f"{name}.mp4").write_bytes(b"x")


def test_the_concat_list_names_each_scene_file_in_order():
    assert concat_lines(("A", "B")) == "file 'A.mp4'\nfile 'B.mp4'\n"


def test_a_missing_ffmpeg_exits_2(media: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(render.shutil, "which", lambda _: None)
    assert render.main() == 2


def test_a_failing_render_exits_1_and_joins_nothing(media: Path, monkeypatch: pytest.MonkeyPatch):
    _runs(monkeypatch, 1)
    assert render.main() == 1
    assert not render.OUTPUT.exists()


def test_an_unfinished_scene_exits_1(media: Path, monkeypatch: pytest.MonkeyPatch):
    def partly() -> None:
        _write_scenes(media)
        (media / f"{render.ORDER[-1]}_temp.mp4").write_bytes(b"x")

    _runs(monkeypatch, partly)
    assert render.main() == 1


def test_a_join_timeout_exits_1(media: Path, monkeypatch: pytest.MonkeyPatch):
    _runs(monkeypatch, lambda: _write_scenes(media), subprocess.TimeoutExpired("ffmpeg", 1))
    assert render.main() == 1
    assert not render.OUTPUT.exists()


def test_a_complete_render_writes_the_output(media: Path, monkeypatch: pytest.MonkeyPatch):
    _runs(monkeypatch, lambda: _write_scenes(media), lambda: render.PARTIAL.write_bytes(b"v"))
    assert render.main() == 0
    assert render.OUTPUT.read_bytes() == b"v"
    assert (media / "concat.txt").read_text(encoding="utf-8") == concat_lines(render.ORDER)
