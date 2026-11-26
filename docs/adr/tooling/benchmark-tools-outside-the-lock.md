---
status: accepted
topic: tooling
created: 2026-11-24
updated: 2026-11-26
---
# Benchmark tools in `bench/<tool>/`, each in a hashed environment outside the lock

## Context

The study scores fills under published benchmark tools beside its own measures
([`docs/RESEARCH.md`](../../RESEARCH.md#published-comparisons)). Such a tool brings what the
pipeline has no other use for: SynthEval imports LightGBM, whose macOS wheel loads Homebrew's
`libomp` at import. SDNist 2.4, the only release on PyPI, pins scipy 1.10.1, which needs Python
below 3.12; SDNist 2.5, tag `v2.5.0` on GitHub with no wheel, states Python 3.9 to 3.12, and its
pins have Python 3.12 wheels ([PyPI metadata](https://pypi.org/pypi/sdnist/json) and
[the tag](https://github.com/usnistgov/SDNist/tree/v2.5.0), read 2026-11-26). The project's
environment is one lock, installed exactly by `make setup`, type-checked and covered by the gate,
which needs no system library and runs on Python 3.14. ManimGL already runs outside that lock in
a hashed environment under `video/` (ADR [`tooling/video-scenes-outside-the-lock`](video-scenes-outside-the-lock.md)).

## Decision Drivers

* **Must keep `make check` runnable on a machine with nothing but the interpreter and the lock.**
* **Must install the same tool on every machine**, from hashes.
* **Must keep `bko`'s side of a benchmark reading in the gate**: the export of its inputs, the
  call and the parsing of what the tool returns.
* **Should give every benchmark tool one layout**, so a second tool costs no new decision.

## Decision

A tool joins the lock as an extra only when the gate can import it with nothing but the
interpreter and the lock. A tool that needs a system library, a different Python, or packages the
gate has no other use for runs in a hashed environment outside the lock, as ManimGL does.

- **Layout.** `bench/<tool>/` holds `requirements.in`, a hashed `requirements.txt`, one script
  that imports no `bko`, the script's own test and a README. `requirements.txt` is compiled with
  `uv pip compile bench/<tool>/requirements.in --universal --python-version 3.14 --generate-hashes -o bench/<tool>/requirements.txt`
  (another Python where the tool needs one).
- **Make targets.** `make <tool>-env` runs `uv venv bench/<tool>/.venv`, then
  `uv pip sync --require-hashes --only-binary :all:` from `requirements.txt`, as `make video-env`
  does; `make <tool>-test` runs the script's test with that environment's interpreter. Neither is
  part of the gate.
- **The call.** `bko` runs the script as a subprocess through an injectable runner, with a
  timeout and an environment of `PATH`, `HOME`, the locale and thread caps only, so no credential
  reaches the tool. It exports the script's inputs to a temporary directory, logs each input's
  sha256, and refuses with exit 2, naming `make <tool>-env`, when the environment is absent.
- **The tests.** `bko`'s side of a reading is tested in the gate without the tool; the script's
  real test runs under `make <tool>-test`, and a PR that changes the script says it ran.
- **The gate's reach.** The gate lints `bench/` and holds it to the size cap; it neither
  type-checks nor covers it.

## Rationale

An extra in `uv.lock` is installed by `make setup` on every machine, so a tool that needs a system
library would make the gate fail where the library is absent, and a tool on another Python cannot
share the lock at all. A hashed environment per tool keeps the lock and the gate's machine
requirements as they are and still installs the same packages everywhere. Running the script as a
subprocess keeps the tool out of `bko`'s imports, so `bko` stays typed and covered, and its side
of the reading can be tested without the tool; the script's real test is the one step the gate
cannot run, so it has its own Make target.

## Consequences

**Gained**: the lock, `make setup` and the gate's requirements do not change when a benchmark tool
arrives; each tool installs from hashes; every tool reads its inputs the same way.
**Accepted**: a benchmark script is neither type-checked nor covered, and its real test runs only
when someone runs `make <tool>-test`. Each tool's environment must be rebuilt when its
`requirements.txt` changes.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A locked extra for SynthEval | Every machine that runs `make check` would need `brew install libomp`, because LightGBM loads it at import on macOS | 2026-11-24 |
| `make setup` installing Homebrew packages | The gate needs no system library today, and `make setup` installs only from the lock | 2026-11-24 |
| One environment shared by every benchmark tool | SDNist 2.4 needs Python below 3.12 and SDNist 2.5 Python 3.9 to 3.12, while SynthEval resolves on 3.14 | 2026-11-24 |

## Enforcement

- Owner: `Makefile` (`syntheval-env`, `syntheval-test`), `tools/check_file_size.py`,
  `bench/syntheval/requirements.txt`, `src/bko/evaluate/syntheval_reading.py`
- Pinned by: `tests/test_repo_gate.py` (the `syntheval-env` and `syntheval-test` recipes),
  `tests/test_file_size.py` (the `bench/` cap, `CAPS` in `tools/check_file_size.py`; ruff's
  configuration reaches `bench/`), `tests/test_evaluate_syntheval.py` (a fake runner for the
  export, the call and the parsing; the runner's environment and timeout; the missing
  environment; a failing script; the script parsed with `ast`, its imports and its flags against
  `score_argv`); outside the gate, `bench/syntheval/tests/test_score.py` under
  `make syntheval-test`

## Reversal Conditions

**LightGBM's macOS wheel ships its own OpenMP runtime**: `python -c "import lightgbm"` succeeds on
a machine with no `libomp`. SynthEval could then join the lock as an extra under ADR
[`tooling/uv-ruff-ty`](uv-ruff-ty.md).

## Related

- [Demo video scenes in `video/`, with ManimGL outside the lock](video-scenes-outside-the-lock.md)
- [uv, ruff and ty](uv-ruff-ty.md)
