"""Rewrite the golden files `tests/test_evaluate_waves.py` compares the scoring harness with.

Run from anywhere: `uv run --no-sync python tests/golden/waves/regenerate.py`.
"""

import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tests.test_evaluate_waves import _run  # ruff: ignore[module-import-not-at-top-of-file]


with tempfile.TemporaryDirectory() as tmp:
    scores, candidates, calls = _run(Path(tmp), in_flight=1)
(HERE / "scores.csv").write_bytes(scores)
(HERE / "candidates.jsonl").write_bytes(candidates)
(HERE / "calls.jsonl").write_bytes(calls)
