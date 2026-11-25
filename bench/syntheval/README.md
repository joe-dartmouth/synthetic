# `bench/syntheval/`: SynthEval outside the lock

SynthEval 1.7.3 scores a synthetic table against real and holdout data. It runs here, in its own
hashed environment, because LightGBM, which it imports, needs Homebrew's `libomp` on macOS (ADR
[`tooling/benchmark-tools-outside-the-lock`](../../docs/adr/tooling/benchmark-tools-outside-the-lock.md)).

- `requirements.in` names SynthEval; `requirements.txt` is compiled from it with hashes.
- `score.py` scores one synthetic table and writes each metric's value as one JSON object. It
  imports no `bko`; `python -m bko.evaluate syntheval` runs it as a subprocess (ADR
  [`research/syntheval-reading`](../../docs/adr/research/syntheval-reading.md)).
- `tests/test_score.py` runs `score.py` under the real SynthEval on a tiny fixture fill.

`make syntheval-env` builds `.venv` here and `make syntheval-test` runs the test; neither is part
of the gate ([`docs/DEVELOPMENT.md`](../../docs/DEVELOPMENT.md#benchmark-environments)).
