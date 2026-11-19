# video: the demo video

ManimGL scenes that show the coverage-gap problem, what a BKO is, and how the pipeline finds and
fills gaps. The labels on screen are short and there is no audio: a presenter talks over the video
and pauses it. [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#video) holds the install, the render
command and where the output lands.

`model.py` holds the running example, built from `bko.bkb` objects. `captions.py` holds the scene
order, the label text and the figures taken from `docs/EVIDENCE.md`. `layout.py` and `sections/`
hold the scenes. `render.py` is the driver that renders each scene and joins them.

One scene's last frame, as a still under `video/media/stills/`:

```
PYTHONPATH=src:. video/.venv/bin/python -m manimlib video/main.py <Scene> -s -w --config_file video/config.yml --video_dir video/media/stills
```

`make lint` and `make sizes` reach `video/`; `make typecheck`, coverage and the docs tests do not,
and the gate never renders.

- **Dependencies:** ManimGL, outside the lock, and FFmpeg.
- **Status:** built.
