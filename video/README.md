# video: the demo video

ManimGL scenes that show the coverage-gap problem, what a BKO is, and how the pipeline finds and
fills gaps and is evaluated; they state no experiment result. Each beat shows a short label and a
one-line explainer, and there is no audio: a presenter talks over the video and pauses it.
[`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#video) holds the install, the render command and
where the output lands.

`model.py` holds the running example, built from `bko.bkb` objects. `captions.py` holds the scene
order, each beat's label and explainer, and the facts the example takes from the docs. `layout.py`
holds the scene frame and the diagram pieces the sections share, and `sections/` one Scene class per
section. `render.py` is the driver that renders each scene and joins them.

One scene's last frame, as a still under `video/media/stills/`:

```
PYTHONPATH=src:. video/.venv/bin/python -m manimlib video/main.py <Scene> -s -w --config_file video/config.yml --video_dir video/media/stills
```

`make lint` and `make sizes` reach `video/`; `make typecheck` reports nothing in it, and coverage
and the docs tests do not reach it. The `tests/test_video_*.py` files hold the model, the
captions' facts, the driver's exit codes and the scene order; the gate never renders.

- **Dependencies:** ManimGL, outside the lock, and FFmpeg.
- **Status:** built.
