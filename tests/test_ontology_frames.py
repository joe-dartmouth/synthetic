"""`bko.ontology.frames` imports no `bko.evaluate` module at import time, so `bko.gaps` can read
every frame, and `check_records` holds records to the frame it is given."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.datasets.frames import FrameSpec, check_records
from bko.datasets.uci import UCI_FRAME
from bko.ontology.frames import DIABETES_FRAME
from tests._corpus import REPO_ROOT


def _record(frame: FrameSpec) -> pd.DataFrame:
    """One record of `frame`: each coded column's first code, the label 0, every other field 0."""
    row = {name: next(iter(frame.codebook.get(name, [0]))) for name in frame.columns}
    row[frame.label] = 0
    return pd.DataFrame([row], columns=list(frame.columns))


def test_gaps_can_import_every_frame() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    script = (
        "import sys\n"
        "import bko.gaps\n"
        "from bko.ontology.frames import (\n"
        "    DIABETES_FIELD_ROWS, DIABETES_FRAME, SYNTHEA_FIELD_ROWS, SYNTHEA_FRAME,\n"
        ")\n"
        "print(sorted(m for m in sys.modules if m == 'bko.evaluate' "
        "or m.startswith('bko.evaluate.')))"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_a_label_outside_the_codebook_is_refused() -> None:
    records = _record(DIABETES_FRAME)
    check_records(records, DIABETES_FRAME)
    records.loc[0, DIABETES_FRAME.label] = 7
    with pytest.raises(ContractError, match=f"{DIABETES_FRAME.label} holds codes \\[7\\]"):
        check_records(records, DIABETES_FRAME)


def test_records_are_held_to_the_frame_they_are_given() -> None:
    records = _record(UCI_FRAME)
    check_records(records, UCI_FRAME)
    with pytest.raises(ContractError, match="columns"):
        check_records(records, DIABETES_FRAME)
