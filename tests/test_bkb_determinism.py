"""Fused BKBs and their queries come out the same under every string-hash seed."""

from __future__ import annotations

import os
import subprocess
import sys

from tests._corpus import REPO_ROOT


DRIVER = """
import hashlib
import sys

from bko.bkb.fusion import fuse
from bko.bkb.reason import log_joint_complete
from bko.bkb.serialize import render
from tests._bkb_fixtures import PIRATE, RELIABILITIES, exported, exported_queries

fragments = [exported(name) for name in PIRATE]
query = next(q for q in exported_queries() if q.bkb == "pirate_fused_r1")
for collapse in (False, True):
    fused = fuse(fragments, list(RELIABILITIES), collapse=collapse)
    sys.stdout.write(hashlib.sha256(render(fused).encode()).hexdigest() + "\\n")
    result = log_joint_complete(fused, query.target, query.evidence)
    pairs = [(s, None if v is None else v.hex()) for s, v in sorted(result.items())]
    sys.stdout.write(repr(pairs) + "\\n")
"""


def run_driver(seed: str) -> str:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", DRIVER],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_fusion_and_queries_do_not_depend_on_the_hash_seed():
    outputs = {run_driver(seed) for seed in ("0", "1", "2", "4294967295")}
    assert len(outputs) == 1
    assert len(outputs.pop().splitlines()) == 4
