"""Rewrite the golden files `tests/test_evaluate_waves.py` compares the scoring harness with.

Run from anywhere: `uv run --no-sync python tests/golden/waves/regenerate.py`.
"""

import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    """Run the harness at in-flight 1 and write each golden file beside this script."""
    sys.path.insert(0, str(HERE.parents[2]))
    from tests.test_evaluate_waves import FILES, _run

    with tempfile.TemporaryDirectory() as tmp:
        made = _run(Path(tmp), in_flight=1)
    for name, data in zip(FILES, made, strict=True):
        (HERE / name).write_bytes(data)


if __name__ == "__main__":
    main()
