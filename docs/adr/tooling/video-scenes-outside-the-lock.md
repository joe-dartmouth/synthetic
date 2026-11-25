---
status: accepted
topic: tooling
created: 2026-11-18
updated: 2026-11-24
---
# Demo video scenes in `video/`, with ManimGL outside the lock

## Context

The team shows its system with a short animated video drawn with ManimGL (3b1b/manim). ManimGL
brings OpenGL, Pango and FFmpeg bindings that no stage of the pipeline needs. The project's
environment is one lock, installed exactly by `make setup`, type-checked and covered by the gate. A
precedent exists for a tool kept outside that lock: the pybkb oracle runs in a throwaway
environment built from a hashed requirements file.

## Decision Drivers

* **Must leave `uv.lock`, `.venv` and every gate verdict the same on every machine.**
* **Must install the same ManimGL on every machine**, from hashes.
* **Should keep the scene code linted and size-capped** like the rest of the tracked Python.

## Decision

We will keep the scenes in a top-level `video/` directory and install ManimGL outside the lock.

- `video/` is its own tree with an entry in `CAPS` (`tools/check_file_size.py`).
- ManimGL 1.7.2 and its dependencies are listed with hashes in `video/requirements.txt`, compiled
  from `video/requirements.in` for Python 3.14, and installed into `video/.venv` by
  `make video-env` (`uv venv`, then `uv pip sync --require-hashes --only-binary :all:`).
  `make video` renders with that environment's interpreter.
- The gate lints and size-caps `video/`. It neither type-checks nor covers it, and it never
  renders.
- The AGENTS rule against `uv pip install` concerns the project environment; `video/.venv` is not
  that environment.

## Rationale

An extra in `uv.lock` would be installed by `make sync`, so `ty`'s verdict would depend on whether a
machine ran it, and a version fork between ManimGL's pins and the project's would make
`make synced` refuse the environment. A separate hashed environment keeps the lock untouched and
still installs the same packages everywhere, as the pybkb oracle does. A tree outside `src/` keeps
ManimGL out of the typed, covered package while `CAPS` and ruff still reach it.

## Consequences

**Gained**: the lock, the project environment and the gate do not change; the scenes are linted
and size-capped.
**Accepted**: the scenes are neither type-checked nor covered, so the numbers they show live in
`video/model.py` and `video/captions.py`, which import no ManimGL and are tested in the gate. A
second environment must be rebuilt when `video/requirements.txt` changes.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A `video` extra in `uv.lock` | `make sync` would install it, making `ty`'s verdict machine-dependent, and a version fork makes `make synced` exit 2 | 2026-11-18 |
| `uv run --with` | Hash enforcement unconfirmed | 2026-11-18 |
| `uvx` | Hash enforcement unconfirmed | 2026-11-18 |
| `src/bko/video` | Typed and covered, and pulls ManimGL into `make setup` (ADR [`tooling/uv-ruff-ty`](uv-ruff-ty.md)) | 2026-11-18 |
| `presentations/` | Costs the same `CAPS` entry and makes the deck folder a Python package | 2026-11-18 |
| ManimGL `master` | A months-old wgpu renderer with open Metal validation reports, and a git requirement cannot carry hashes | 2026-11-18 |

## Enforcement

- Owner: `Makefile` (`video-env`, `video`) and `video/requirements.txt`
- Pinned by: `tests/test_file_size.py`, `tests/test_local_artifacts.py`, `tests/test_video_model.py`

## Reversal Conditions

**Something under `src/`, `tests/` or `tools/` imports ManimGL**:
`grep -rnE "^\s*(from|import) manimlib" src tests tools` prints a line.

**ManimGL co-resolves inside the lock**: `uv pip compile` shows manimgl resolving with the
project's dependencies with no package at two versions.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
- [Per-tree line caps on Python files](file-size-cap.md)
- [Benchmark tools in `bench/<tool>/`, outside the lock](benchmark-tools-outside-the-lock.md)
- [BKB layer source](../research/bkb-layer-source.md)
