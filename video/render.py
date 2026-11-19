"""Render every scene of the demo video and join them into `video/media/bko-demo.mp4`.

Run as `python -m video.render` from `video/.venv` (`make video`). Each scene of
`video.captions.ORDER` is rendered to its own file by ManimGL, then FFmpeg joins the files with a
stream copy. Exit 2 when FFmpeg or ManimGL is missing, 1 when a scene or the join fails; nothing
is joined unless every scene's file is complete.
"""

import importlib.util
import logging
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from video.captions import ORDER


_LOG = logging.getLogger("video.render")

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
MEDIA: Final = REPO_ROOT / "video" / "media"
SCENES: Final = MEDIA / "scenes"
OUTPUT: Final = MEDIA / "bko-demo.mp4"
PARTIAL: Final = MEDIA / "bko-demo.part.mp4"
RENDER_TIMEOUT_SECONDS: Final = 3600
JOIN_TIMEOUT_SECONDS: Final = 600


def concat_lines(order: Sequence[str]) -> str:
    """FFmpeg's concat list naming each scene's file, in `order`."""
    return "".join(f"file '{name}.mp4'\n" for name in order)


def _render_scenes() -> int:
    """Render every scene of `ORDER` into `SCENES`; 0 when each file is complete, else 1."""
    environment = {**os.environ, "PYTHONPATH": f"src{os.pathsep}."}
    command = [
        sys.executable,
        "-m",
        "manimlib",
        "video/main.py",
        "-a",
        "-w",
        "--config_file",
        "video/config.yml",
        "--video_dir",
        str(SCENES),
    ]
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=RENDER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _LOG.error("rendering the scenes took over %d s", RENDER_TIMEOUT_SECONDS)
        return 1
    if result.returncode != 0:
        _LOG.error("ManimGL exited %d; its output above names the scene", result.returncode)
        return 1
    for name in ORDER:
        if not (SCENES / f"{name}.mp4").is_file() or (SCENES / f"{name}_temp.mp4").exists():
            _LOG.error("scene %s did not finish: no complete %s.mp4", name, name)
            return 1
    return 0


def _join(ffmpeg: str) -> int:
    """Join the scene files into `OUTPUT` through `PARTIAL`; 0 on success, else 1."""
    join = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-i",
        "concat.txt",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(PARTIAL),
    ]
    try:
        (SCENES / "concat.txt").write_text(concat_lines(ORDER), encoding="utf-8")
        joined = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            join, cwd=SCENES, stdin=subprocess.DEVNULL, timeout=JOIN_TIMEOUT_SECONDS, check=False
        )
        if joined.returncode != 0:
            _LOG.error("ffmpeg exited %d joining the scenes", joined.returncode)
            return 1
        PARTIAL.replace(OUTPUT)
    except subprocess.TimeoutExpired:
        _LOG.error("joining the scenes took over %d s", JOIN_TIMEOUT_SECONDS)
        return 1
    except OSError as error:
        _LOG.error("joining the scenes failed: %s", error)
        return 1
    _LOG.info("wrote %s", OUTPUT)
    return 0


def main() -> int:
    """Check the tools, clear the last render, render every scene and join them.

    Returns:
        0 on success, 1 when a scene or the join fails, 2 when FFmpeg or ManimGL is missing.
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _LOG.error("no runnable ffmpeg on PATH; ManimGL writes video through it")
        return 2
    if importlib.util.find_spec("manimlib") is None:
        _LOG.error("manimlib is not importable; run make video-env")
        return 2
    if SCENES.exists():
        shutil.rmtree(SCENES)
    OUTPUT.unlink(missing_ok=True)
    PARTIAL.unlink(missing_ok=True)
    return _render_scenes() or _join(ffmpeg)


if __name__ == "__main__":
    sys.exit(main())
